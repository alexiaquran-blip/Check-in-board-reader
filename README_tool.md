# Attendance Board Tool V3

## Run locally
python attendance_board_tool_v3.py image.png --out run1

## Output files
- run1_results.json
- run1_source_mesh_overlay.png
- run1_piecewise_overlay.png
- run1_piecewise_warp.png

## Build Windows EXE locally
pip install -r requirements.txt
pip install pyinstaller
pyinstaller --clean --noconfirm attendance_board_tool_v3.spec

## Build Windows EXE online with GitHub Actions
1. Push this folder to GitHub
2. Open Actions > Build Windows EXE
3. Run workflow
4. Download the artifact: attendance-board-tool.exe
