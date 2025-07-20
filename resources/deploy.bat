@echo off

set INKSCAPE_PATH="C:\Program Files\Inkscape\bin\inkscape.exe"

@REM Deploying images and icons
copy Sigima.svg ..\sigima\data\logo
copy Sigima-Title.svg ..\doc\_static
%INKSCAPE_PATH% "Sigima-Frontpage.svg" -o "..\doc\_static\Sigima-Frontpage.png" -w 1300
%INKSCAPE_PATH% "Sigima-Banner.svg" -o "..\doc\images\Sigima-Banner.png" -w 364

@REM Generating icon
call :generate_icon "Sigima"

goto:eof

:generate_icon
set ICON_NAME=%1
for %%s in (16 24 32 48 128 256) do (
  %INKSCAPE_PATH% "%ICON_NAME%.svg" -o "tmp-%%s.png" -w %%s -h %%s
)
magick "tmp-*.png" "%ICON_NAME%.ico"
del "tmp-*.png"
goto:eof
