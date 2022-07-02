Paper: Vision for Prosthesis Control Using Unsupervised Labeling of Training Data- https://ieeexplore.ieee.org/abstract/document/9555789.

This contains the code for clustering gait into different movement clusters, and then using those clusters as pseudolabels to train a terrain classification vision system. Drop the subject directories (named xUD***) with processed data (available on https://figshare.com/s/06ef299a0cd56a5a998c) in ./processed/unstructured/. Then run gait_steps_dataprep_v3.m to extract gait cycles from the data, followed by cluster_kinematics.py to create movement clusters. Then run classify_cluster_vision_v2.3_train.py to train the visual classifier and classify_cluster_vision_v2.3_infer.py for inference.

(left) Clustered gait types identified in https://ieeexplore.ieee.org/abstract/document/9555789. Depending on how low in the hierarchy, the data could be
divided into flat ground vs. stair walking, or further subdivided into turning gait, straight walking, and stair ascent and
descent. (right) Mean knee angles for each of the clusters. The kinematic trajectories are distinct for different kinds of
ambulation, leading to clusters that differentiate ambulation modes.

![raiDendrogramAndKneeAngles](https://user-images.githubusercontent.com/42185229/176980707-aca42244-b244-4534-bcce-a68b1cac97a9.png)
