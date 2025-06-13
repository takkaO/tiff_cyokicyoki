import cv2
import os
from tkinter import filedialog
import numpy as np
from PIL import Image
import tifffile
import imagecodecs._imcd
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class TiffRectangleSelector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_tiff = None
        self.display_image = None
        self.scale_factor = 1.0
        
        # TIFF画像を読み込み
        self.load_tiff_image()
        
        self.clone = self.display_image.copy()
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.current_rectangle = None  # 現在の矩形（表示座標系）
        self.current_original_rectangle = None  # 現在の矩形（元画像座標系）
        self.current_mouse_pos = None  # 現在のマウス位置
        self.show_crosshair = True  # クロスヘア表示フラグ
        
    def load_tiff_image(self):
        """TIFF画像を読み込んで表示用に変換"""
        try:
            # まずtifffileで読み込み
            self.original_tiff = tifffile.imread(self.image_path)
            print(f"TIFF画像情報:")
            print(f"- 形状: {self.original_tiff.shape}")
            print(f"- データ型: {self.original_tiff.dtype}")
            print(f"- 値の範囲: {self.original_tiff.min()} - {self.original_tiff.max()}")
            
            # 表示用に8ビットに変換
            display_array = self.convert_for_display(self.original_tiff)
            
            # OpenCV形式に変換（BGR）
            if len(display_array.shape) == 3:
                if display_array.shape[2] == 3:  # RGB
                    self.display_image = cv2.cvtColor(display_array, cv2.COLOR_RGB2BGR)
                elif display_array.shape[2] == 4:  # RGBA
                    self.display_image = cv2.cvtColor(display_array, cv2.COLOR_RGBA2BGR)
                else:
                    self.display_image = display_array
            else:  # グレースケール
                self.display_image = cv2.cvtColor(display_array, cv2.COLOR_GRAY2BGR)
                
            # 画像が大きすぎる場合はリサイズ
            height, width = self.display_image.shape[:2]
            max_display_size = 800
            
            if max(height, width) > max_display_size:
                if width > height:
                    new_width = max_display_size
                    new_height = int(height * max_display_size / width)
                else:
                    new_height = max_display_size
                    new_width = int(width * max_display_size / height)
                
                self.scale_factor = min(new_width / width, new_height / height)
                self.display_image = cv2.resize(self.display_image, (new_width, new_height))
                print(f"表示用にリサイズ: {width}x{height} -> {new_width}x{new_height} (scale: {self.scale_factor:.3f})")
            
        except Exception as e:
            print(f"tifffileでの読み込みエラー: {e}")
            # PILで試行
            try:
                pil_image = Image.open(self.image_path)
                print(f"PIL TIFF画像情報:")
                print(f"- モード: {pil_image.mode}")
                print(f"- サイズ: {pil_image.size}")
                
                # numpy配列に変換
                self.original_tiff = np.array(pil_image)
                display_array = self.convert_for_display(self.original_tiff)
                
                if len(display_array.shape) == 3:
                    self.display_image = cv2.cvtColor(display_array, cv2.COLOR_RGB2BGR)
                else:
                    self.display_image = cv2.cvtColor(display_array, cv2.COLOR_GRAY2BGR)
                    
            except Exception as pil_error:
                raise ValueError(f"TIFF画像の読み込みに失敗しました: tifffile={e}, PIL={pil_error}")
    
    def convert_for_display(self, array):
        """配列を8ビット表示用に正規化（色再現性を向上）"""
        return array
        # if array.dtype == np.uint8:
        #     return array
        # elif array.dtype == np.uint16:
        #     # 16ビット -> 8ビット（より正確な変換）
        #     # 実際の値の範囲を考慮して変換
        #     if array.max() <= 255:
        #         # 8ビット範囲内の場合はそのまま
        #         return array.astype(np.uint8)
        #     else:
        #         # 16ビットの上位8ビットを使用（従来の/256より正確）
        #         return (array >> 8).astype(np.uint8)
        # elif array.dtype == np.uint32:
        #     # 32ビット -> 8ビット
        #     if array.max() <= 255:
        #         return array.astype(np.uint8)
        #     elif array.max() <= 65535:
        #         return (array >> 8).astype(np.uint8)
        #     else:
        #         return (array >> 24).astype(np.uint8)
        # elif array.dtype in [np.float32, np.float64]:
        #     # 浮動小数点の場合、実際の値の範囲を考慮
        #     if array.min() >= 0 and array.max() <= 1:
        #         # 0-1の正規化済みの場合
        #         return (array * 255).astype(np.uint8)
        #     else:
        #         # その他の浮動小数点は最小最大正規化
        #         if array.max() > array.min():
        #             normalized = (array - array.min()) / (array.max() - array.min())
        #             return (normalized * 255).astype(np.uint8)
        #         else:
        #             return np.zeros_like(array, dtype=np.uint8)
        # else:
        #     # その他の型は最小最大正規化
        #     if array.max() > array.min():
        #         normalized = (array - array.min()) / (array.max() - array.min())
        #         return (normalized * 255).astype(np.uint8)
        #     else:
        #         return np.zeros_like(array, dtype=np.uint8)
    
    def display_to_original_coords(self, point):
        """表示座標を元画像座標に変換"""
        if self.scale_factor != 1.0:
            return (int(point[0] / self.scale_factor), int(point[1] / self.scale_factor))
        return point
    
    def draw_crosshair(self, image, x, y):
        """マウス周辺に小さなクロスヘアを描画"""
        if not self.show_crosshair:
            return
            
        # クロスヘアのサイズ（片側の長さ）
        crosshair_size = 15
        
        # クロスヘアの色（白と黒の2重線で視認性向上）
        white_color = (255, 255, 255)
        black_color = (0, 0, 0)
        
        # 縦線（上下）
        cv2.line(image, (x, y - crosshair_size), (x, y + crosshair_size), black_color, 3)  # 太い黒線
        cv2.line(image, (x, y - crosshair_size), (x, y + crosshair_size), white_color, 1)  # 細い白線
        
        # 横線（左右）
        cv2.line(image, (x - crosshair_size, y), (x + crosshair_size, y), black_color, 3)  # 太い黒線
        cv2.line(image, (x - crosshair_size, y), (x + crosshair_size, y), white_color, 1)  # 細い白線
        
        # 中央の小さな円
        cv2.circle(image, (x, y), 3, black_color, -1)
        cv2.circle(image, (x, y), 2, white_color, -1)
    
    def draw_mouse_info(self, image, x, y):
        """マウス座標情報を描画"""
        # 元画像座標を計算
        orig_coords = self.display_to_original_coords((x, y))
        
        # 情報文字列を作成
        info_text = f"Display: ({x}, {y}) | Original: ({orig_coords[0]}, {orig_coords[1]})"
        
        # テキストサイズを取得
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(info_text, font, font_scale, thickness)
        
        # 背景矩形を描画
        padding = 5
        bg_start = (5, 5)
        bg_end = (text_width + padding * 2, text_height + padding * 2)
        cv2.rectangle(image, bg_start, bg_end, (0, 0, 0), -1)  # 黒背景
        cv2.rectangle(image, bg_start, bg_end, (255, 255, 255), 1)  # 白枠
        
        # テキストを描画
        text_pos = (padding + 5, text_height + padding)
        cv2.putText(image, info_text, text_pos, font, font_scale, (255, 255, 255), thickness)

    def mouse_callback(self, event, x, y, flags, param):
        # マウス位置を更新
        self.current_mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            # 常に画像を更新してクロスヘアを表示
            self.display_image = self.clone.copy()
            
            # 現在の矩形があれば再描画
            if self.current_rectangle:
                cv2.rectangle(self.display_image, self.current_rectangle[0], self.current_rectangle[1], (0, 255, 0), 2)
            
            if self.drawing:
                # 現在ドラッグ中の矩形を描画
                cv2.rectangle(self.display_image, self.start_point, (x, y), (0, 0, 255), 2)
            
            # クロスヘアを描画
            self.draw_crosshair(self.display_image, x, y)
            
            # マウス座標情報を表示
            self.draw_mouse_info(self.display_image, x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.end_point = (x, y)
                
                # 矩形のサイズが十分大きい場合のみ保存
                if abs(self.end_point[0] - self.start_point[0]) > 5 and \
                   abs(self.end_point[1] - self.start_point[1]) > 5:
                    
                    # 新しい矩形で前の矩形を置き換え
                    self.current_rectangle = (self.start_point, self.end_point)
                    
                    # 元画像座標系での矩形を計算
                    orig_start = self.display_to_original_coords(self.start_point)
                    orig_end = self.display_to_original_coords(self.end_point)
                    self.current_original_rectangle = (orig_start, orig_end)
                    
                    print(f"矩形が選択されました:")
                    print(f"  表示座標: {self.start_point} -> {self.end_point}")
                    print(f"  元画像座標: {orig_start} -> {orig_end}")
                    print(f"  元画像サイズ: {abs(orig_end[0] - orig_start[0])} x {abs(orig_end[1] - orig_start[1])}")
                
                # 最終的な画像を更新
                self.display_image = self.clone.copy()
                if self.current_rectangle:
                    cv2.rectangle(self.display_image, self.current_rectangle[0], self.current_rectangle[1], (0, 255, 0), 2)
                
                # クロスヘアも再描画
                if self.current_mouse_pos:
                    self.draw_crosshair(self.display_image, self.current_mouse_pos[0], self.current_mouse_pos[1])
                    self.draw_mouse_info(self.display_image, self.current_mouse_pos[0], self.current_mouse_pos[1])
    
    def extract_region(self):
        """現在選択されている矩形領域を元のTIFF画像から切り出し"""
        if not self.current_original_rectangle:
            return None
            
        start, end = self.current_original_rectangle
        
        # 座標を正規化（左上と右下を確定）
        x1, y1 = min(start[0], end[0]), min(start[1], end[1])
        x2, y2 = max(start[0], end[0]), max(start[1], end[1])
        
        # 境界チェック
        height, width = self.original_tiff.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        # 領域を切り出し
        if len(self.original_tiff.shape) == 3:
            return self.original_tiff[y1:y2, x1:x2, :]
        else:
            return self.original_tiff[y1:y2, x1:x2]
    
    def save_extracted_region(self, output_path="extracted_region.tif"):
        """選択した領域をTIFFファイルとして保存"""
        region = self.extract_region()
        if region is not None:
            tifffile.imwrite(output_path, region)
            print(f"選択領域を保存: {output_path} (shape: {region.shape})")
            return True
        return False
    
    def run(self):
        cv2.namedWindow('TIFF Rectangle Selector', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('TIFF Rectangle Selector', self.mouse_callback)
        
        print("\n操作方法:")
        print("- マウスをドラッグして矩形を選択（新しい矩形で前の矩形を置き換え）")
        print("- 'Enter'キーでトリミング処理を実行")
        print("   => 別フォルダに出力され、元ファイルは変化しません")
        print("- 'r'キーで矩形をクリア")
        print("- 'ESC'キーまたは'q'キーで終了")
        if self.scale_factor != 1.0:
            print(f"- 注意: 表示用にリサイズされています (scale: {self.scale_factor:.3f})")
        
        window_name = 'TIFF Rectangle Selector'
        while True:
            cv2.imshow(window_name, self.display_image)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q') or key == 13:  # ESCまたはqで終了
                break
            elif key == ord('r'):  # rでリセット
                self.current_rectangle = None
                self.current_original_rectangle = None
                self.display_image = self.clone.copy()
                if self.current_mouse_pos and self.show_crosshair:
                    self.draw_crosshair(self.display_image, self.current_mouse_pos[0], self.current_mouse_pos[1])
                    self.draw_mouse_info(self.display_image, self.current_mouse_pos[0], self.current_mouse_pos[1])
                print("矩形をクリアしました")
            # elif key == ord('s'):  # sで保存
            #     if self.current_original_rectangle:
            #         self.save_extracted_region()
            #     else:
            #         print("保存する矩形がありません")
            # elif key == ord('c'):  # cでクロスヘア切り替え
            #     self.show_crosshair = not self.show_crosshair
            #     print(f"クロスヘア表示: {'ON' if self.show_crosshair else 'OFF'}")
            #     # 画像を更新
            #     self.display_image = self.clone.copy()
            #     if self.current_rectangle:
            #         cv2.rectangle(self.display_image, self.current_rectangle[0], self.current_rectangle[1], (0, 255, 0), 2)
            #     if self.current_mouse_pos and self.show_crosshair:
            #         self.draw_crosshair(self.display_image, self.current_mouse_pos[0], self.current_mouse_pos[1])
            #         self.draw_mouse_info(self.display_image, self.current_mouse_pos[0], self.current_mouse_pos[1])

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                # Xボタンで閉じる
                break

        cv2.destroyAllWindows()
        if key == 13:
            return self.current_original_rectangle
        else:
            return None

def get_image_path_list(target_dir_path = None):
    if target_dir_path is None:
        os.getcwd()
    return sorted(glob.glob("*.tif", root_dir = target_dir_path))

def crop_and_save(target_dir_path, rectangles, fname):
    try:   
        tiff = tifffile.imread(os.path.join(target_dir_path, fname))
        crop_tiff = None

        start, end = rectangles        
        # 座標を正規化（左上と右下を確定）
        x1, y1 = min(start[0], end[0]), min(start[1], end[1])
        x2, y2 = max(start[0], end[0]), max(start[1], end[1])
        
        # 境界チェック
        height, width = tiff.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
    
        # 領域を切り出し
        if len(tiff.shape) == 3:
            crop_tiff = tiff[y1:y2, x1:x2, :]
        else:
            crop_tiff = tiff[y1:y2, x1:x2]

        if crop_tiff is None:
            print(f"Error {os.path.join(target_dir_path, fname)}")
        
        tifffile.imwrite(os.path.join(target_dir_path, "crop_result", fname), crop_tiff)
    except Exception as e:
        error_log = f"Error: {fname}\n"
        error_log += f"=> {e}\n"
        return error_log
    return ""

def main():
    target_dir_path = filedialog.askdirectory(
        initialdir = os.getcwd(),
        title = "TIFF 読み込みフォルダ選択",
        mustexist = True
        ) 
    tif_list = get_image_path_list(target_dir_path)
    if len(tif_list) < 1:
        print("処理対象ファイルが見つかりませんでした")
        print("終了するには何かキーを押してください。")
        input()
        return
    # 使用例
    tiff_name = tif_list[0]  # TIFFファイルのパスを指定
    
    try:
        selector = TiffRectangleSelector(os.path.join(target_dir_path, tiff_name))
        rectangles = selector.run()

        if rectangles is None:
            return
        
        os.makedirs(os.path.join(target_dir_path, "crop_result"), exist_ok=True)
        print(f"保存先：{os.path.join(target_dir_path, 'crop_result')}")

        error_log = ""

        with ThreadPoolExecutor(max_workers=4) as executor:
            # 全てのタスクを一度にサブミット
            future_to_fname = {
                executor.submit(crop_and_save, target_dir_path, rectangles, fname): fname 
                for fname in tif_list
            }

            # 完了したタスクから順番に結果を取得
            with tqdm(total=len(tif_list), desc="Cropping", ncols=80, colour='green') as pbar:
                for future in as_completed(future_to_fname):
                    fname = future_to_fname[future]
                    try:
                        error_log += future.result()
                    except Exception as e:
                        print(f"Unexpected error for {fname}: {e}")
                    finally:
                        pbar.update(1)

        if not error_log == "":
            print(error_log)
            print("終了するには何かキーを押してください。")
            input()
            
    except ValueError as e:
        print(f"エラー: {e}")
        print("終了するには何かキーを押してください。")
        input()

if __name__ == "__main__":
    print("初期化処理中...（しばらくお待ちください）")
    main()