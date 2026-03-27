# Inference Pipeline for CapAware models

## Usage for Handover Prediction model:
1. Build
```
docker build -t birkan-handover-prediction . -f Dockerfile-handover-prediction
```
2. Run for each 5G router (only for 5G SA)

```
docker run -d \
  --gpus all \
  --env router=CAU-R16-4312 \
  --restart unless-stopped \
  birkan-handover-prediction
```

```

docker run -d \
  --gpus all \
  --env router=CAU-R16-4329 \
  --restart unless-stopped \
  birkan-handover-prediction
```

```
docker run -d \
  --gpus all \
  --env router=5G-D2-WAVELAB \
  --restart unless-stopped \
  birkan-handover-prediction
```









## Usage for Bandwidth Prediction model:
1. Build
```
docker build -t birkan-bandwidth-prediction . -f Dockerfile-bandwidth-prediction
```
2. Run for each 5G router (only for 5G SA + 5G NR bands n28 (FDD + 1x1 MIMO UL), n3 (FDD + 1x1 MIMO UL), and n78 (TDD + 2x2 MIMO UL))

```
docker run -d \
  --gpus all \
  --name CAU-R16-4312 \
  --env router=CAU-R16-4312 \
  --restart unless-stopped \
  birkan-bandwidth-prediction
```

```
docker run -d \
  --gpus all \
  --name CAU-R16-4329 \
  --env router=CAU-R16-4329 \
  --restart unless-stopped \
  birkan-bandwidth-prediction
```

```
docker run -d \
  --gpus all \
  --name HAW-Stecknitz \
  --env router=HAW-Stecknitz \
  --restart unless-stopped \
  birkan-bandwidth-prediction
```

```
docker run -d \
  --gpus all \
  --name HAW-Nobiskrug \
  --env router=HAW-Nobiskrug \
  --restart unless-stopped \
  birkan-bandwidth-prediction
```

```
docker run -d \
  --gpus all \
  --name 5G-D2-WAVELAB \
  --env router=5G-D2-WAVELAB \
  --restart unless-stopped \
  birkan-bandwidth-prediction
```