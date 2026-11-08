# RealWaste dataset attribution

The images in `plastic/`, `metal/`, `paper/`, `general trash/`, and the copied test examples come from **RealWaste**, by **Sam Single, Saeid Iranmanesh, and Raad Raad (2023)**.

- Original source: https://github.com/sam-single/realwaste
- Dataset citation: Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G
- Associated paper: *RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning* (Information, 2023), linked from the original source.

The original GitHub source states that its dataset is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**: https://creativecommons.org/licenses/by-nc-sa/4.0/

This repository retains that notice for the redistributed source images. It includes four of the original nine categories (2,706 images); `Miscellaneous Trash` maps to the local folder `general trash` and the output label `Trash`. The original image bytes are retained. Training preprocessing resizes and augments images in memory; it does not alter the source image files. Example images are copies from the fixed held-out partition.

This notice applies to the dataset images; it does not assign a license to ORBIT source code.
