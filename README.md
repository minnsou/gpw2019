## Overview

GPW 2019用のレポジトリです。麻雀におけるpolicy function関連の研究に用いたコードをおいています、

abstractディレクトリには、extended abstractの実験のコードとその結果を載せています。

それ以外のディレクトリやファイルは、それ以降の研究の結果です。

## Description

abstractフォルダには3層MLPの実験、Gaoらの再現実験、築地らの再現実験、Residual Blockの実験の4つのコードとその結果を置いています。

1. 3層MLPの実験 コード：mini_mahjong_pnsplot_split.py 結果：result3.out result3.png  
1. Gaoらの再現実験 コードと結果：mini_mahjong_PNplot_gaoCNN.ipynb  
1. 築地らの再現実験 コードと結果：mini_mahjong_PNplot_tsukijiCNN.ipynb  
1. Residual Blockの実験 コード：mini_mahjong_pnplot_resnet.py 結果：result_resnet_3.out result_resnet_3.png

## Requirement

keras, numpy, pandas, matplotlib, sklearn

## Usage

`python mini_mahjong_pnplot_split.py > result3.out`

`python mini_mahjong_pnplot_resnet.py > result_resnet_3.out`

ipynbファイルはそのまま開けばコードと結果を見ることができます。上から順に実行していってください。