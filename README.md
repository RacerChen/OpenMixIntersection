# OpenMixIntersection

> A open source dataset and model for autonomous vehicles. The detailed description will be updated after our paper is accepted. Keep waiting, please... ; )

## Project Structure

|-OpenMixIntersection

​	|-AL_ML

​		|-Datasets: Open source dataset, divided into features and tags in .csv files

​		|-Log: Training logs

​		|-Model: Trained final model

​		|-Tool: Tools to convert datasets

​		|-Train: Train codes

​	|-Simulations

​		|-ConfigFiles: Files to construct simulation environments

​		|-SubSystem: Codes for realizing the simulation and generating datasets

​	|-README

## Datasets' Format Introduction

![data_format](https://github.com/RacerChen/OpenMixIntersection/tree/main/img)

| r_n  | Type   | Meaning |
| ---- | ------ | ------- |
| r1   | ONEHOT | xxx     |
| r2   | ONEHOT | xxx     |
| r3   | float  | xxx     |
| r4   | float  | xxx     |
| r5   | float  | xxx     |
| r6   | float  | xxx     |
| r7   | float  | xxx     |
| r8   | float  | xxx     |
| r9   | ONEHOT | xxx     |
| r10  | float  | xxx     |
| r11  | float  | xxx     |

| b_n  | Type   | Meaning |
| ---- | ------ | ------- |
| b1   | float  | xxx     |
| b2   | float  | xxx     |
| b3   | float  | xxx     |
| b4   | ONEHOT | xxx     |
| b5   | ONEHOT | xxx     |
| b6   | ONEHOT | xxx     |

| p_n  | Type   | Meaning |
| ---- | ------ | ------- |
| p1   | float  | xxx     |
| p2   | ONEHOT | xxx     |
| p3   | float  | xxx     |

> The meaning xxx will be described after our paper is accepted. Keep waiting, please... ; )

