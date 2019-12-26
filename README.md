## Overview

GPW 2019用のレポジトリです。麻雀におけるpolicy function関連の研究に用いたコードやその結果などを置いています。

abstractディレクトリは、extended abstractの実験のコードとその結果を載せています。

paperのディレクトリは、論文の実験のコードとその結果を載せています。

この研究で扱ったミニ麻雀（1人麻雀）をUnityで作りました。[ここ]()で公開します。（リンク先は準備中）

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

paperフォルダには、教師あり学習を行ったmini_mahjong_SL、強化学習を行ったmini_mahjong_RL、その他の実験を行ったmini_mahjong_utilsがあります。mini_mahjong_SLの中にあるsimpleフォルダは役なし条件で、yakuフォルダは役あり条件です。


## Requirement

tensorflow, keras, numpy, pandas, matplotlib, sklearn, optuna, openai baselines, openai gym

## Usage

`python mini_mahjong_pnplot_split.py`

基本的にはどのpyファイルも上記のように実行するだけですが、paperフォルダのmini_mahjong_RLディレクトリだけは少し異なります。

mahjong_env.pyは一人麻雀の環境を作っており、mahjong_utils.pyは麻雀の役判定やあがり判定などをおいているだけで、mahjong_networks.pyはQ関数を構築しているだけなので、直接実行はしません。学習をさせる時にtrain.pyを実行し、その実行結果と理論値を比較するにはeval_model.pyを実行してください。層の数などのネットワークや手牌の枚数などの麻雀の条件を変えるには、train.pyもeval_model.pyもオプションを付けて変更します。詳しくは
`python train.py -h`
を見てください。

ipynbファイルはそのまま開けばコードと結果を見ることができます。上から順に実行していってください。

## Paper

本論文は[こちら](https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=199990&item_no=1&page_id=13&block_id=8)（リンク先は情報処理学会）

ポスターは[poster.pdf](https://github.com/minnsou/gpw2019/blob/master/poster.pdf)をご覧ください。

### 本論文のお詫びと訂正

上記論文に誤りがありましたので、お詫びして訂正いたします。

1つ目 p3 4.教師あり学習　2段落目

（誤）役がある時とない時で最善手が変わる手牌は 11385 個のうち...

（正）役がある時とない時で最善手が変わる手牌は 10389 個のうち...


2つ目 p3 4.教師あり学習　2段落目

（誤）その価値の差がもっとも大きい手牌は 11222334 で...役なしにおける最善手は3で...役ありにおける最善手は4で...

（正）その価値の差がもっとも大きい手牌は 22333445 で...役なしにおける最善手は4で...役ありにおける最善手は5で...

3つ目 p5 表1 最終行

（誤）caoら

（正）Gaoら

それに伴い、表現を少し改めた修正版を[ここ](http://www.tanaka.ecc.u-tokyo.ac.jp/wp/tshimizu/2019/11/14/gpw2019%e8%ab%96%e6%96%87%e3%81%ae%e8%a8%82%e6%ad%a3/)に掲載しておきます。
