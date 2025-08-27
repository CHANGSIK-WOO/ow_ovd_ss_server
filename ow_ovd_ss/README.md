# OW-OVD: Unified Open World and Open Vocabulary Object Detection

## Abstract
Open world perception expands traditional closed-set frameworks, which assume a predefined set of known categories, to encompass dynamic real-world environments. Open World Object Detection (OWOD) and Open Vocabulary Object Detection (OVD) are two main research directions, each addressing unique challenges in dynamic environments. However, existing studies often focus on only one of these tasks, leaving the combined challenges of OWOD and OVD largely underexplored. In this paper, we propose a novel detector, OW-OVD, which inherits the zero-shot generalization capability of OVD detectors while incorporating the ability to actively detect unknown objects and progressively optimize performance through incremental learning, as seen in OWOD detectors. To achieve this, we start with a standard OVD detector and adapt it for OWOD tasks. For attribute selection, we propose the Visual Similarity Attribute Selection (VSAS) method, which identifies the most generalizable attributes by computing similarity distributions across annotated and unannotated regions. Additionally, to ensure the diversity of attributes, we incorporate a similarity constraint in the iterative process. Finally, to preserve the standard inference process of OVD, we propose the Hybrid Attribute-Uncertainty Fusion (HAUF) method. This method combines attribute similarity with known class uncertainty to infer the likelihood of an object belonging to an unknown class. We validated the effectiveness of OW-OVD through evaluations on two OWOD benchmarks, M-OWODB and S-OWODB. The results demonstrate that OW-OVD outperforms existing state-of-the-art models, achieving a +15.3 improvement in unknown object recall (U-Recall) and a +15.5 increase in unknown class average precision (U-mAP).

## Pipeline
![alt text](assets/method.png)

## Results
### U-Recall
![alt text](assets/u_recall.png)

### U-mAP
![alt text](assets/u_map.png)

## Citations
Our code is based on YOLO_World. If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝.

```bibtex
@inproceedings{Cheng2024YOLOWorld,
  title={YOLO-World: Real-Time Open-Vocabulary Object Detection},
  author={Cheng, Tianheng and Song, Lin and Ge, Yixiao and Liu, Wenyu and Wang, Xinggang and Shan, Ying},
  booktitle={Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```