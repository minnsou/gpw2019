## Overview

GPW 2019用のレポジトリです。麻雀におけるpolicy function関連の研究に用いたコードをおいています。

abstractディレクトリは、extended abstractの実験のコードとその結果を載せています。

paperのディレクトリは、論文の実験のコードとその結果を載せています。

## Description

abstractフォルダには3層MLPの実験、Gaoらの再現実験、築地らの再現実験、Residual Blockの実験の4つのコードとその結果を置いています。

1. 3層MLPの実験
   - コード：mini_mahjong_pnsplot_split.py
   - 結果：result3.outとresult3.png  
1. Gaoらの再現実験
   - コードと結果：mini_mahjong_PNplot_gaoCNN.ipynb  
1. 築地らの再現実験
   - コードと結果：mini_mahjong_PNplot_tsukijiCNN.ipynb  
1. Residual Blockの実験
   - コード：mini_mahjong_pnplot_resnet.py
   - 結果：result_resnet_3.outとresult_resnet_3.png

paperフォルダには、教師あり学習を行ったmini_mahjong_SL、強化学習を行ったmini_mahjong_RL、その他の実験を行ったmini_mahjong_utilsがあります。教師あり学習の中にあるsimpleフォルダは役なし条件で、yakuフォルダは役あり条件です。


## Requirement

keras, numpy, pandas, matplotlib, sklearn, optuna, openai baselines, openai gym, tensorflow

## Usage

`python mini_mahjong_pnplot_split.py`

基本的には上記のように実行するだけです。paperのmini_mahjong_RLだけはtrain_test.pyを実行するようにしてください。

ipynbファイルはそのまま開けばコードと結果を見ることができます。上から順に実行していってください。

実験条件の細かい変更に関してはコードを見て自分で変えてください。
