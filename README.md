# D2NN 鑷敱绌洪棿鍏ㄨ繛鎺ュ厜瀛︾缁忕綉缁滀豢鐪?

褰撳墠浠ｇ爜瀹炵幇浜嗗彲楠岃瘉鐨勪豢鐪熷唴鏍革紝榛樿灞傞棿璺濅负 5 mm锛屽苟鍙仈鍚堜紭鍖栫浉浣嶅拰鎸箙锛?

- 澶嶆暟瑙掕氨浼犳挱锛?
- 涓績闆跺～鍏呬笌瑁佸壀锛?
- 5 灞傜浉浣?鎸箙鍙井鍓嶅悜妯″瀷锛?
- 10 涓煩褰㈡帰娴嬪尯鍩熺殑绉垎鍏夊己璁＄畻锛?
- 鍒嗙被銆佹眹鑱氭晥鐜?eta銆佸姣斿害锛?
- NumPy 鍙傝€冨疄鐜板拰 PyTorch 鍙井瀹炵幇锛?
- 瑙掕氨鑳介噺瀹堟亽銆佹帰娴嬪尯銆丯umPy/PyTorch 涓€鑷存€у拰姊害鍐掔儫娴嬭瘯銆?

## 鐞嗚淇

鏂规鏂囨。绗?4 鑺傚皢绗?k 涓帰娴嬪尯鍩熺殑娴嬮噺閲忓畾涔変负绉垎鍏夊己锛?

`sk = sum_{(x,y) in Dk} |Udet(x,y)|^2`

绗?7 鑺傛帹瀵兼搴︽椂鎶婂悓涓€閲忓啓鎴愬尯鍩熷唴澶嶆尟骞呬箣鍜岀殑妯℃柟 `|sum U|^2`銆備袱鑰呭彧鏈夊湪鍖哄煙鍐呮墍鏈夊鎸箙鍚岀浉鏃舵墠杩戜技涓€鑷达紝涓€鑸帰娴嬪櫒娴嬮噺鐨勬槸閫愮偣鍏夊己绉垎銆傚洜姝ゆ湰浠ｇ爜榛樿閲囩敤锛?

`sk = sum |U|^2`

姊害浣跨敤 PyTorch autograd 浠庤鐗╃悊瀹氫箟鐩存帴璁＄畻銆?

## 鏈湴杩愯鐜

宸查獙璇佸彲鐢ㄧ殑鐜锛?

- Python 3.8.2锛?
- torch 2.2.2+cu121锛?
- torchvision 0.17.2+cu121锛?
- NumPy 1.24.4锛?
- SciPy 1.10.1锛?
- GPU锛歂VIDIA GeForce RTX 3060 Laptop锛? GB锛?
- PyTorch CUDA 鍙敤銆?

杩愯锛?

```powershell
& 'E:\codes\env\STME-main\Scripts\python.exe' -m unittest discover -s tests -v
& 'E:\codes\env\STME-main\Scripts\python.exe' scripts\smoke_test.py
& 'E:\codes\env\STME-main\Scripts\python.exe' scripts\train.py --dataset mnist --data-root data --phase-init zero --padded-size 4096 --layer-spacing-mm 5 --detector-spacing-mm 10 --limit 8 --max-batches 2 --micro-batch-size 2 --effective-batch-size 4 --epochs 1 --device cuda
```

## 鐩綍

- `src/d2nn/`锛氱墿鐞嗘牳銆佹帰娴嬪櫒銆丯umPy 鍜?PyTorch 妯″瀷锛?
- `tests/`锛氭暟鍊煎拰姊害楠岃瘉锛?
- `scripts/`锛氱鍒扮鍐掔儫娴嬭瘯銆佹樉瀛樺熀鍑嗗拰璁粌鍏ュ彛锛?
- `docs/`锛氭柟妗堝垎鏋愩€佺悊璁轰慨姝ｃ€侀浂濉厖鏀舵暃銆丮NIST 棣栬疆璁粌鍜?5 mm/鎸箙浼樺寲鍙樻洿銆?

## 濉厖绛栫暐

- `Npad=2048`锛氶€熷害浼樺厛鐨勫巻鍙茶繎浼煎€硷紝鏈夐檺瀛斿緞澶氬眰绾х粨鏋滀笉鍙俊锛?
- `Npad=4096`锛? mm 灞傞棿璺濅笅鐨勫綋鍓嶆湰鍦拌缁冮厤缃紝micro-batch=2锛?
- `Npad=6144`锛氬彲閫夐珮绮惧害鍙傝€冮厤缃紝micro-batch=1锛?
- 瀹屾暣鎺ㄥ鍜屽疄娴嬭 `docs/闆跺～鍏呮敹鏁涚粨璁?md`銆?

## 鐩镐綅涓庢尟骞呬紭鍖?

璁粌鍏ュ彛榛樿鍚屾椂浼樺寲鐩镐綅鍜屾尟骞呫€傛尟骞呴€氳繃 sigmoid 绾︽潫鍦ㄧ墿鐞嗛€忚繃鐜囪寖鍥?`(0, 1)`銆傚鍔?`--phase-only` 鍙仮澶嶇函鐩镐綅妯″紡銆?

## 璁粌缁撴灉

- [5 mm 灞傞棿璺濈浉浣?鎸箙璁粌缁撴灉](docs/5mm鐩镐綅鎸箙璁粌缁撴灉.md)

## 版本保存

每次修改都会创建新的 Git commit 并推送到 GitHub，不覆盖已发布历史。完整版本记录见 [VERSIONS.md](VERSIONS.md)。

