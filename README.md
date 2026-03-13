# Phan loai vien thuoc tu hinh anh (CNN + Transfer Learning)

Do an nay duoc xay dung theo de tai:
- Phan loai vien thuoc tu hinh anh bang CNN
- Dataset dinh huong: ePillID-FEWSHOT-metrix
- Mo hinh: ResNet-50, EfficientNet-B0, Vision Transformer (ViT-B/16)
- Bai toan: Multi-class classification voi transfer learning
- Giao dien: Tkinter

## 1. Cau truc du an

- run_gui.py: mo phan mem giao dien Tkinter
- train_cli.py: train model
- src/train.py: pipeline huan luyen
- src/inference.py: pipeline suy luan + doi chieu anh mau
- src/features.py: so khop 4 dac diem (mau sac, hinh dang, kich thuoc, texture)
- src/gui_tk.py: giao dien nguoi dung

## 2. Yeu cau giao dien

Giao dien co tieu de "Phan loai thuoc bang anh" va bang 4 cot:
- Ten thuoc
- Hinh thuoc mau
- Anh muon kiem tra (nguoi dung tai len)
- Ket qua

Quy tac ket qua:
- Dung (True) neu trung >= 3/4 dac diem
- Sai (False) neu duoi nguong nay

## 3. Dinh dang dataset

Du an mong cho dataset theo cau truc ImageFolder:

```
dataset_root/
  train/
    class_0/
    class_1/
    ...
  val/
    class_0/
    class_1/
    ...
  test/
    class_0/
    class_1/
    ...
```

Mac dinh GUI dang tro toi:
- ../VAIPE/data/epillid_split_debug
- Mapping ten thuoc: ../VAIPE/data/mapping_standard.json

## 4. Cai dat

```bash
pip install -r requirements.txt
```

## 5. Train model

Vi du train bang ResNet-50:

```bash
python train_cli.py --data-root ../VAIPE/data/epillid_split_debug --model resnet50 --epochs 5 --batch-size 8 --out models/resnet50_best.pt
```

Doi model:
- --model efficientnet_b0
- --model vit_b_16

Neu may khong co Internet de tai pretrain:

```bash
python train_cli.py --data-root ../VAIPE/data/epillid_split_debug --model resnet50 --no-pretrained
```

## 6. Chay phan mem

```bash
python run_gui.py
```

Trong GUI:
1. Chon file model .pt
2. Chon dataset root
3. Chon mapping json (neu co)
4. Bam "Nap model"
5. Bam "Chon anh can kiem tra"

Phan mem se:
- Du doan lop thuoc
- Lay anh mau trong dataset
- So khop 4 dac diem
- Tra ve Dung/Sai

## 7. Luu y

- Day la phien ban do an huong ung dung, uu tien tinh de su dung va kha nang demo.
- Do chinh xac phu thuoc vao chat luong du lieu train.
