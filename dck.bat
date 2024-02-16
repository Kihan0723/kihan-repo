@echo off
setlocal enabledelayedexpansion

set "folder_path=C:\TRNSYS18\dck"
set "executable_path=C:\TRNSYS18\Exe\trnEXE64.exe"

for %%f in ("%folder_path%\*.dck") do (
    start / wait "" "!executable_path!" "/h" "%%f"
    
    )

endlocal