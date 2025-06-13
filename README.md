TIFF CyokiCyoki

TIFF 画像一括トリミングツール

## Prepare

```sh
py -m pip install -r requirements.txt
py main.py
```

## Build

```sh
create-version-file ./version.yml
pyinstaller ./main.py --onefile --clean --name TIFF_CyokiCyoki --version-file ./version_file.txt --icon icon.ico
```