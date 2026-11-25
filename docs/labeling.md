# Labelling your in-the-wild set (Roboflow)

Goal: 200–300 Indian road images that are not from RDD2022. Most become a held-out test set for generalisation (about 70%), and the rest are added to training (about 30%).

## 1. Collect
- Use dashcam or phone footage mounted at windshield height, facing forward, like RDD2022.
- Vary the conditions on purpose: day and night, wet and dry, highway and city. Tag these in Roboflow so Phase 4 can break results down by them.
- Extract frames sparsely, at most 1 frame per 2 s of driving. This avoids near-duplicates:
  ```
  ffmpeg -i drive01.mp4 -vf fps=0.5 -q:v 2 drive01__%04d.jpg
  ```
- **Naming is required:** `<video-id>__<frame>.jpg`, with two underscores. Photos also need a group id, e.g. `phone-mgroad__0001.jpg`. The merge script keeps each video entirely in train or entirely in test.

## 2. Label in Roboflow
1. Create a project with type *Object Detection*.
2. Make exactly these classes, spelled exactly like this:
   `longitudinal_crack`, `transverse_crack`, `alligator_crack`, `pothole`.
3. Box rules (match RDD2022):
   - **longitudinal_crack (D00):** a crack running along the direction of travel. Box the whole visible length.
   - **transverse_crack (D10):** a crack running across the lane.
   - **alligator_crack (D20):** interconnected cracking in a network or block pattern. Box the whole patch.
   - **pothole (D40):** a bowl-shaped hole. Box the rim.
   - Skip damage that's too far away to identify, and skip manholes, road markings and patches.
   - Keep images with no damage, with no boxes. They are useful negatives.
4. Do **not** add Roboflow augmentations or preprocessing; set resize to *none*. Augmentation happens at training time.

## 3. Export and merge
1. Export as *YOLOv8*, download the zip, and unzip it anywhere.
2. Run:
   ```
   python -m src.data.merge_custom path/to/export
   ```
   This writes `data/splits/custom_train.txt`, `data/splits/wild_test.txt` and `results/custom_report.json`.
