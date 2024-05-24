训练ID3模型 start
训练ID3模型 success,training_time=0.023100852966308594
Prediction Time: 0.0000001631 seconds per sample
Accuracy: 0.86
Precision: 0.83
F1 Score: 0.84
Recall: 0.86
Confusion Matrix:
[[10236   295]
 [ 1376   341]]
 AUC for class 0: 0.73
AUC for class 1: 0.73
Feature: new_user, Importance: 0.03718220208881804
Feature: age, Importance: 0.22297329587249934
Feature: sex, Importance: 0.03692280476437814
Feature: source, Importance: 0.06463401147124398
Feature: total_pages_visited, Importance: 0.6382876858030605


训练C45模型 start
给定第(0, 1, 2, 3, 4)特征为连续值特征
训练C45模型 success,training_time=2.1177785396575928
Prediction Time: 0.0000055709 seconds per sample
Accuracy: 0.87
Precision: 0.85
F1 Score: 0.84
Recall: 0.87
Confusion Matrix:
[[10379   152]
 [ 1408   309]]

训练CART模型 start
训练CART模型 success,training_time=0.025120973587036133
Prediction Time: 0.0000001631 seconds per sample
Accuracy: 0.86
Precision: 0.83
F1 Score: 0.84
Recall: 0.86
Confusion Matrix:
[[10227   304]
 [ 1374   343]]
AUC for class 0: 0.73
AUC for class 1: 0.73
Feature: new_user, Importance: 0.053327792960292886
Feature: age, Importance: 0.23615046543384371
Feature: sex, Importance: 0.04385676202758169
Feature: source, Importance: 0.06526013388579323
Feature: total_pages_visited, Importance: 0.6014048456924884




| 指标\算法                           | ID3                                | C4.5                              | CART                               |
| ------------------------------ | ---------------------------------- | --------------------------------- | ---------------------------------- |
| 总训练时间(秒)trainingTime     | 0.0231                             | 2.1177                            | 0.0251                             |
| 每次预测时间(秒)predictionTime | 0.16e-06                           | 5.6e-6                            | 0.16e-06                           |
| 准确度 accuracy                | 0.86                               | 0.87                              | 0.86                               |
| 精度                           | 0.83                               | 0.85                              | 0.83                               |
| F1                             | 0.84                               | 0.84                              | 0.84                               |
| 召回率                         | 0.86                               | 0.87                              | 0.86                               |
| matrix                         | [ [10236   295 ] , [ 1376   341] ] | [ [10379   152] , [ 1408   309] ] | [ [10227   304 ] , [ 1374   343] ] |


| 特征\算法                | ID3    | CART   |
| ------------------- | ------ | ------ |
| new_user            | 0.0372 | 0.0533 |
| age                 | 0.2230 | 0.2362 |
| sex                 | 0.0369 | 0.0439 |
| source              | 0.0646 | 0.0653 |
| total_pages_visited | 0.6383 | 0.6014 |

