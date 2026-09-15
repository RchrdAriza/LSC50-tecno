# LSC50-tecno

Colombian Sign Language (LSC) motion-capture research dataset: inertial (IMU) + OpenSim kinematics + MediaPipe landmark tracks across 5 volunteers.

## Layout

- `IMU/` — per-volunteer inertial and biomechanics data (`RAW`, `STO`, `OUT_OPENSIM`, `INFO`).
- `LANDMARKS/` — MediaPipe landmark tracks (`BODY_LANDMARKS`, `FACE_LANDMARKS`, `HANDS_LANDMARKS`), 1000 CSVs each.

See `AGENTS.md` for detailed data semantics and format gotchas.

## Dataset

This project uses the **LSC50: Colombian Sign Language Video and Inertial Measurement Dataset**:

> Flórez-Sierra, A. F., Solórzano, B. D., Segura-Quijano, F., Cortés-Bello, Y. M., Cubillos, L., Giraldo, L. F., & Cifuentes-De la Portilla, C. (2024). *LSC50: Colombian Sign Language Video and Inertial Measurement dataset*. Scientific Data, 11, 1347.
> https://doi.org/10.1038/s41597-024-04172-5

The dataset was developed by the authors cited above and is used in this project for research and development purposes.