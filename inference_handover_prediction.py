import os
import torch
import json
import joblib
import numpy as np
import paho.mqtt.client as mqtt
from collections import deque

import models
from mqtt_config import MQTT_username, MQTT_password, MQTT_host, MQTT_port

# Put PyTorch on the first GPU, a.k.a. RTX 3080 on wyk
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model checkpoint
model_checkpoint_version = "19vu2hl8"
model = models.CapAwareHandoverPredictor.load_from_checkpoint(f"./Handover-Prediction/{model_checkpoint_version}/checkpoints/epoch=89-step=1130400-val_loss=0.119.ckpt")

model = model.to(device)
model.eval()  # Set to evaluation mode

# Load the scaler
try:
    scaler_label = joblib.load(f'scaler-save/prediction_handover/{model_checkpoint_version}/scaler_label.gz')
    scaler_input = joblib.load(f'scaler-save/prediction_handover/{model_checkpoint_version}/scaler_feature.gz')
    print("Scaler files loaded successfully!")
except (FileNotFoundError, Exception) as e:
    print(f"Missing or invalid scaler files! Error: {e}")
    exit(1) # Stop execution if scalers are missing

def process_data(data):
    rsrp = float(data['lte']['lRsrp'])
    sinr = float(data['lte']['lSinr'])
    cqi = float(data['lte']['lCqi'])
    
    speed_str = data['gps'].get('speed', '0')
    speed = float(speed_str.split()[0]) if isinstance(speed_str, str) else float(speed_str)
    
    # Must match the expected feature order: ['SPEED', 'RSRP', 'SINR', 'CQI']
    return np.array([speed, rsrp, sinr, cqi], dtype=np.float32)

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
        processed_data = process_data(data)
        buffer.append(processed_data)

        # Make predictions only if the buffer is full
        if len(buffer) == BUFFER_LENGHT:
            transformed_buffer = scaler_input.transform(buffer)

            input_tensor = torch.tensor(transformed_buffer, dtype=torch.float32)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(device)  # Move input tensor to the same device as the model

            with torch.no_grad():
                logits = model(input_tensor)
                prediction = torch.sigmoid(logits)

            # Convert to real values
            pred_array = prediction.cpu().numpy() # Ensure correct shape for inverse transform

            handover_probability = float(pred_array[0, 0])   # single value if pred_len == 1
            handover_probability = round(handover_probability, 3)

            float_RSRP = float(data['lte']['lRsrp'])

            # Mirrors MikroTik UI
            if float_RSRP >= -80:
                RSRP_status = 'Excellent'
            elif float_RSRP >= -90:
                RSRP_status = 'Good'
            elif float_RSRP >= -100:
                RSRP_status = 'Fair'
            elif float_RSRP < -100:
                RSRP_status = 'Poor'
            else:
                RSRP_status = 'Undefined'
            
            float_current_cell_ID = float(data['lte']['lCurrentCellid'])

            result = {
                'identity': data['identity'],
                'time': data['time'],
                'version': model_checkpoint_version,
                'operator': data['lte']['lCurrentOperator'],
                'handover_probability': handover_probability,
                'RSRP': float_RSRP,
                'RSRP_status' : RSRP_status,
                'current_cell_ID' : float_current_cell_ID
            }

            result_json = json.dumps(result)
            client.publish(f"captnfoerdeareal/prediction/wan-handover/{result['identity']}", result_json, qos=1)
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
