import os
import torch
import json
import joblib
import numpy as np
import paho.mqtt.client as mqtt
from collections import deque
from sklearn.preprocessing import OneHotEncoder

import models
from mqtt_config import MQTT_username, MQTT_password, MQTT_host, MQTT_port

# Put PyTorch on the second GPU, a.k.a. RTX 4060 on wyk
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# Making sure that we use CUDA
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

cwd = os.getcwd()

# Returns default router if the key doesn't exist
# CAU-R16-4312
# 5G-D2-WAVELAB
router = os.environ.get('router', 'CAU-R16-4312')

# Determine the device (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model checkpoint
model_checkpoint_version = "e24eeda9"
model = models.CapAwareBandwidthPredictor.load_from_checkpoint(f"./Uplink-Bandwidth-Prediction/{model_checkpoint_version}/checkpoints/epoch=10-step=11803-val_loss=0.095.ckpt")

model = model.to(device)
model.eval()  # Set to evaluation mode

# Load the scaler
try:
    scaler_label = joblib.load(f'scaler-save/prediction_bandwidth/{model_checkpoint_version}/scaler_label.gz')
    scaler_input = joblib.load(f'scaler-save/prediction_bandwidth/{model_checkpoint_version}/scaler_input.gz')
    print("Scaler files loaded successfully!")
except (FileNotFoundError, Exception) as e:
    print(f"Missing or invalid scaler files! Error: {e}")
    exit(1) # Stop execution if scalers are missing

# Create OneHotEncoder instance and fit it on the master list of bands
master_bands = ['n28', 'n3', 'n78']  # Master list of bands
ohe = OneHotEncoder(categories=[master_bands],  # same master list
                    drop='first',            # matches drop_first=True
                    dtype=np.float32,
                    handle_unknown='ignore',
                    sparse_output=False)

ohe.fit(np.array(master_bands).reshape(-1, 1))  # ← single “offline” fit

def process_data(data):
    sinr = float(data['lte']['lSinr'])
    cqi = float(data['lte']['lCqi'])
    rsrp = float(data['lte']['lRsrp'])

    basic_band_name = data['lte']['lPrimaryBand'].split('@')[0]

    # hacky workaround to map n1 to n3
    # n1 is not in the master list of bands in the training data, but n3 is
    # they have similar characteristics (e.g., FDD)
    # n3 goes into the model
    # in the prediction part below, we only check if the live incoming data has n1 and adjust the prediction accordingly
    if basic_band_name == 'n1':
        basic_band_name = 'n3'

    ohe_encoded_band = ohe.transform([[basic_band_name]])  # shape (1, n_cols)

    signal_strengths = np.array([sinr, cqi, rsrp], dtype=np.float32)

    input = np.concatenate((signal_strengths, ohe_encoded_band), axis=None)

    return input

# MQTT Configuration
TOPIC = f'captnfoerdeareal/wan/{router}/#'
#CLIENT_ID = "mqtt-pytorch-client"

# Data Buffer for Last N Timestamps
BUFFER_LENGHT=16 # Number of timestamps the model expects
buffer = deque(maxlen=BUFFER_LENGHT)

# MQTT Callbacks
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(TOPIC)
    else:
        print(f"Failed to connect, return code {reason_code}")

def on_message(client, userdata, msg):
    try:
        # Parse and preprocess the data
        data = json.loads(msg.payload.decode('utf-8'))

        # make sure the connection is 5G SA
        # the model is only trained on 5G SA data
        if data['lte']['lDataClass'] == '5G SA':
            print(f"Received 5G SA data: {data}")
            processed_data = process_data(data)
            buffer.append(processed_data)
        else:
            print(f"Received non-5G SA data: {data}")
            # clear buffer and start over when we receive non-5G SA data
            buffer.clear()
            return

        # Make predictions only if the buffer is full
        if len(buffer) == BUFFER_LENGHT:
            transformed_buffer = scaler_input.transform(buffer)

            input_tensor = torch.tensor(transformed_buffer, dtype=torch.float32)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(device)  # Move input tensor to the same device as the model

            with torch.no_grad():
                prediction = model(input_tensor)
        
            # Convert to real values
            pred_array = prediction.cpu().numpy()

            real_values = scaler_label.inverse_transform(pred_array[0])
            
            bandwidth = float(real_values[0][0]) / (1000 * 1000)
            

            # hacky workaround to adjust predictions for Telekom 5G SA with n78 at 90 MHz
            # training data was for Vodafone 5G SA with n78 at 80 MHz
            if data['identity'] == 'CAU-R16-4329' and data['lte']['lPrimaryBand'].split('@')[0] == 'n78':
                # CAU-R16-4329 (Telekom 5G SA)  uses n78 at 90 MHz with 245 PRBs
                # CAU-R16-4312 (Vodafone 5G SA) uses n78 at 80 MHz with 217 PRBs
                # Thus, we can scale up the prediction by 245/217 = 1.12903226
                bandwidth = bandwidth * (245/217)

            # hacky workaround to adjust predictions for Telekom 5G SA with n1 at 20 MHz
            # training data was for Vodafone 5G SA with n3 at 25 MHz
            elif data['identity'] == 'CAU-R16-4329' and data['lte']['lPrimaryBand'].split('@')[0] == 'n1':
                # CAU-R16-4329 (Telekom 5G SA)  uses n1 at 20 MHz with 106 PRBs
                # CAU-R16-4312 (Vodafone 5G SA) uses n3 at 25 MHz with 133 PRBs
                # Thus, we can scale down the prediction by 106/133 = 1.25471698
                bandwidth = bandwidth * (106/133)

            used_bandwidth = float(data['lte']['ltxbitspersecond']) / (1000 * 1000)

            bandwidth = round(bandwidth, 2)
            used_bandwidth = round(used_bandwidth, 2)
            
            result = {
                'identity': data['identity'],
                'time': data['time'],
                'version': model_checkpoint_version,
                'operator': data['lte']['lCurrentOperator'],
                'predicted_bandwidth': bandwidth,
                'used_bandwidth': used_bandwidth,
            }

            result_json = json.dumps(result)
            client.publish(f"captnfoerdeareal/prediction/wan-uplink-bandwidth/{result['identity']}", result_json, qos=1)
        else:
            print(f"Buffer not full yet ({len(buffer)}/{BUFFER_LENGHT})")
    except Exception as e:
        print(f"Error processing message: {e}")

# MQTT Client Setup
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2) # type: ignore
client.on_connect = on_connect
client.on_message = on_message

# Local Wavelab MQTT server
client.username_pw_set(MQTT_username, MQTT_password)
client.connect(MQTT_host, MQTT_port)

# Start MQTT loop
try:
    client.loop_forever()
except KeyboardInterrupt:
    print("Disconnected from MQTT Broker")
