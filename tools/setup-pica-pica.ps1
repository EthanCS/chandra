param([string]$Blender)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$revision='b433c2edc0c3120b6d61a9c64daa2d24ebc677a9'
$source=Join-Path $root '.local/pica-pica-assets/pica-pica-mini-diorama-01'
New-Item -ItemType Directory -Force $source | Out-Null
$files=@('Mini_Diorama_01.fbx','Caution_Decal_01_D.tga','Decal_Keyboard_01_D.tga','First_Aid_Wall_Kit_01_D.tga','Plastic_04_R.tga','StickyNote_02.tga','Wax_Pastel_Label_Yellow_01.tga')
foreach($name in $files) {
    $path=Join-Path $source $name
    if(!(Test-Path -LiteralPath $path)) {
        Invoke-WebRequest "https://raw.githubusercontent.com/SEED-EA/pica-pica-assets/$revision/pica-pica-mini-diorama-01/$name" -OutFile $path
    }
}
if(!$Blender) {
    $Blender=(Get-ChildItem -LiteralPath "$env:ProgramFiles/Blender Foundation" -Filter blender.exe -Recurse | Sort-Object FullName -Descending | Select-Object -First 1).FullName
}
if(!$Blender -or !(Test-Path -LiteralPath $Blender)){throw 'Blender 4.2 or newer is required. Pass -Blender with its executable path.'}
& $Blender -b --factory-startup --python-exit-code 1 --python (Join-Path $PSScriptRoot 'import-pica-pica.py') -- $source (Join-Path $root 'external/PicaPica')
if($LASTEXITCODE -ne 0){throw 'PICA conversion failed.'}
