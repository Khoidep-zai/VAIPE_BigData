from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from src.inference import infer_with_verification, load_predictor


class PillApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Phan loai thuoc bang anh")
        self.geometry("1180x720")
        self.configure(bg="#f4f7fb")

        self.predictor = None
        self.current_query_photo = None
        self.current_sample_photo = None

        self.checkpoint_var = tk.StringVar(value="models/best_model.pt")
        self.dataset_var = tk.StringVar(value="../VAIPE/data/epillid_split_debug")
        self.mapping_var = tk.StringVar(value="../VAIPE/data/mapping_standard.json")

        self.drug_name_var = tk.StringVar(value="-")
        self.result_var = tk.StringVar(value="-")
        self.conf_var = tk.StringVar(value="-")
        self.details_var = tk.StringVar(value="")

        self._build_ui()

    def _build_ui(self) -> None:
        title = tk.Label(
            self,
            text="Phan loai thuoc bang anh",
            font=("Segoe UI", 24, "bold"),
            bg="#f4f7fb",
            fg="#153049",
        )
        title.pack(pady=12)

        ctrl = tk.Frame(self, bg="#f4f7fb")
        ctrl.pack(fill="x", padx=16)

        self._make_path_row(ctrl, "Model (.pt)", self.checkpoint_var, 0)
        self._make_path_row(ctrl, "Dataset root", self.dataset_var, 1)
        self._make_path_row(ctrl, "Mapping json", self.mapping_var, 2)

        action_bar = tk.Frame(ctrl, bg="#f4f7fb")
        action_bar.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 10))

        ttk.Button(action_bar, text="Nap model", command=self.load_model).pack(side="left", padx=(0, 8))
        ttk.Button(action_bar, text="Chon anh can kiem tra", command=self.select_query_image).pack(side="left")

        table = tk.Frame(self, bg="#f4f7fb")
        table.pack(fill="both", expand=True, padx=16, pady=8)

        headers = ["Ten thuoc", "Hinh thuoc mau", "Anh muon kiem tra", "Ket qua"]
        for i, text in enumerate(headers):
            cell = tk.Label(
                table,
                text=text,
                font=("Segoe UI", 13, "bold"),
                bg="#dbe8f5",
                fg="#12324a",
                relief="ridge",
                bd=1,
                padx=6,
                pady=8,
            )
            cell.grid(row=0, column=i, sticky="nsew")

        for i, weight in enumerate((2, 3, 3, 2)):
            table.grid_columnconfigure(i, weight=weight)

        name_cell = tk.Label(
            table,
            textvariable=self.drug_name_var,
            font=("Segoe UI", 12),
            bg="white",
            relief="ridge",
            bd=1,
            wraplength=240,
            justify="center",
        )
        name_cell.grid(row=1, column=0, sticky="nsew", ipadx=8, ipady=110)

        self.sample_img_label = tk.Label(table, bg="white", relief="ridge", bd=1)
        self.sample_img_label.grid(row=1, column=1, sticky="nsew", ipadx=8, ipady=8)

        self.query_img_label = tk.Label(table, bg="white", relief="ridge", bd=1)
        self.query_img_label.grid(row=1, column=2, sticky="nsew", ipadx=8, ipady=8)

        result_cell = tk.Frame(table, bg="white", relief="ridge", bd=1)
        result_cell.grid(row=1, column=3, sticky="nsew")

        tk.Label(result_cell, textvariable=self.result_var, font=("Segoe UI", 20, "bold"), fg="#0e5a33", bg="white").pack(pady=(80, 8))
        tk.Label(result_cell, textvariable=self.conf_var, font=("Segoe UI", 12), fg="#294057", bg="white").pack()

        detail = tk.Label(
            self,
            textvariable=self.details_var,
            justify="left",
            anchor="w",
            bg="#eef4fb",
            fg="#253648",
            font=("Consolas", 11),
            padx=10,
            pady=8,
        )
        detail.pack(fill="x", padx=16, pady=(4, 12))

    def _make_path_row(self, parent: tk.Widget, text: str, var: tk.StringVar, row: int) -> None:
        tk.Label(parent, text=text, bg="#f4f7fb", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=3)
        tk.Entry(parent, textvariable=var, width=90).grid(row=row, column=1, sticky="we", pady=3)
        parent.grid_columnconfigure(1, weight=1)

        def browse() -> None:
            if "json" in text.lower():
                path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")])
            elif "model" in text.lower():
                path = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
            else:
                path = filedialog.askdirectory()
            if path:
                var.set(path)

        ttk.Button(parent, text="...", width=4, command=browse).grid(row=row, column=2, padx=(6, 0), pady=3)

    def load_model(self) -> None:
        checkpoint = self.checkpoint_var.get().strip()
        mapping = self.mapping_var.get().strip()
        if not checkpoint or not Path(checkpoint).exists():
            messagebox.showerror("Loi", "Khong tim thay file model (.pt).")
            return

        try:
            self.predictor = load_predictor(checkpoint, mapping_json_path=mapping if Path(mapping).exists() else None)
            messagebox.showinfo("Thong bao", "Nap model thanh cong")
        except Exception as exc:
            messagebox.showerror("Loi", f"Nap model that bai: {exc}")

    def select_query_image(self) -> None:
        if self.predictor is None:
            messagebox.showwarning("Canh bao", "Ban can nap model truoc")
            return

        query_path = filedialog.askopenfilename(
            filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not query_path:
            return

        dataset_root = self.dataset_var.get().strip()
        try:
            result = infer_with_verification(self.predictor, query_path, dataset_root)
            self._update_result_ui(query_path, result)
        except Exception as exc:
            messagebox.showerror("Loi", f"Khong the suy luan: {exc}")

    def _update_result_ui(self, query_path: str, result: dict) -> None:
        self.drug_name_var.set(result["drug_name"])
        conf = result["confidence"] * 100.0
        self.conf_var.set(f"Confidence: {conf:.2f}%")

        verdict_text = "Dung (True)" if result["verdict"] else "Sai (False)"
        self.result_var.set(f"{verdict_text}\nDiem dac diem: {result['score']}/4")

        self.current_query_photo = self._to_photo(query_path)
        self.query_img_label.configure(image=self.current_query_photo)

        if result.get("sample_image"):
            self.current_sample_photo = self._to_photo(result["sample_image"])
            self.sample_img_label.configure(image=self.current_sample_photo)
        else:
            self.sample_img_label.configure(image="", text="Khong co anh mau")

        compare = result.get("compare")
        if compare:
            metrics = compare["metrics"]
            checks = compare["checks"]
            self.details_var.set(
                " | ".join(
                    [
                        f"color={metrics['color']} ({checks['color']})",
                        f"shape={metrics['shape']} ({checks['shape']})",
                        f"size={metrics['size']} ({checks['size']})",
                        f"texture={metrics['texture']} ({checks['texture']})",
                    ]
                )
            )
        else:
            self.details_var.set("Khong tim thay anh mau de doi chieu dac diem.")

    @staticmethod
    def _to_photo(image_path: str) -> ImageTk.PhotoImage:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((300, 260))
        return ImageTk.PhotoImage(image)


def run_gui() -> None:
    app = PillApp()
    app.mainloop()


if __name__ == "__main__":
    run_gui()
