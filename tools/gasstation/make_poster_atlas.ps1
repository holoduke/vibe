# Packs the pavilion's eight screen pictures and the station's posters into ONE picture
# (assets/office/tex/screens.png, texture slot 36): 4096 x 4096 -
#   rows 0..1151:      the screens, 4 x 2 cells of 1024 x 576 (tools/office/screens_src.png, as it was)
#   rows 1152..4094:   the posters, 6 x 3 cells of 682.67 x 981 (tools/gasstation/posters/p00..p17.png, 2:3)
# cube.frag (the display block) and tools/gasstation/make_gasstation.py poster() go by these numbers.
#   powershell -File tools/gasstation/make_poster_atlas.ps1
Add-Type -AssemblyName System.Drawing
$root = Split-Path (Split-Path $PSScriptRoot)
$bmp = New-Object System.Drawing.Bitmap 4096, 4096, ([System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
$g.Clear([System.Drawing.Color]::FromArgb(255, 20, 20, 22))
$s = [System.Drawing.Bitmap]::FromFile("$root/tools/office/screens_src.png")
$g.DrawImage($s, 0, 0, 4096, 1152); $s.Dispose()
for ($i = 0; $i -lt 18; $i++) {
    $f = "$root/tools/gasstation/posters/p{0:d2}.png" -f $i
    if (-not (Test-Path $f)) { continue }
    $p = [System.Drawing.Bitmap]::FromFile($f)
    $x = [single](($i % 6) * 4096.0 / 6.0); $y = [single](1152 + [math]::Floor($i / 6) * 981)
    $g.DrawImage($p, $x, $y, [single](4096.0 / 6.0), [single]981)
    $p.Dispose()
}
$g.Dispose()
$bmp.Save("$root/assets/office/tex/screens.png", [System.Drawing.Imaging.ImageFormat]::Png); $bmp.Dispose()
"atlas written"
