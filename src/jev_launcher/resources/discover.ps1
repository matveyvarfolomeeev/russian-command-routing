$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$entries = New-Object System.Collections.Generic.List[object]
$successful = New-Object System.Collections.Generic.List[string]
$errors = @{}
$shell = New-Object -ComObject WScript.Shell
$namespaceShell = New-Object -ComObject Shell.Application
$registered = @{}
try {
    $appsFolder = $namespaceShell.Namespace('shell:AppsFolder')
    foreach ($item in $appsFolder.Items()) {
        $registered[$item.Path.ToLowerInvariant()] = @{
            target = $item.ExtendedProperty('System.Link.TargetParsingPath')
            arguments = $item.ExtendedProperty('System.Link.Arguments')
        }
    }
} catch { }

foreach ($source in @('start_menu_user', 'start_menu_common')) {
    try {
        $folderKind = if ($source -eq 'start_menu_user') { 'Programs' } else { 'CommonPrograms' }
        $root = [Environment]::GetFolderPath($folderKind)
        if ($root -and (Test-Path -LiteralPath $root)) {
            foreach ($file in Get-ChildItem -LiteralPath $root -Filter '*.lnk' -Recurse -File) {
                try {
                    $shortcut = $shell.CreateShortcut($file.FullName)
                    $appId = ''
                    try {
                        $directory = $namespaceShell.Namespace($file.DirectoryName)
                        $shellItem = $directory.ParseName($file.Name)
                        $appId = $shellItem.ExtendedProperty('System.AppUserModel.ID')
                    } catch { }
                    $entries.Add([ordered]@{
                        name = $file.BaseName
                        source = $source
                        source_key = $source + ':' + $file.FullName.ToLowerInvariant()
                        launch_type = 'shortcut'
                        launch_target = $file.FullName
                        resolved_target = $shortcut.TargetPath
                        raw_arguments = $shortcut.Arguments
                        working_directory = $shortcut.WorkingDirectory
                        icon_target = $shortcut.TargetPath
                        description = $shortcut.Description
                        app_id = $appId
                    })
                    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($shortcut)
                } catch {
                    $errors[$source + ':' + $file.BaseName] = 'Не удалось прочитать ярлык'
                }
            }
        }
        $successful.Add($source)
    } catch {
        $errors[$source] = 'Не удалось прочитать меню Пуск'
    }
}

try {
    foreach ($app in Get-StartApps) {
        $metadata = $registered[$app.AppID.ToLowerInvariant()]
        $entries.Add([ordered]@{
            name = $app.Name
            source = 'start_apps'
            source_key = 'start_apps:' + $app.AppID
            launch_type = 'app_id'
            launch_target = $app.AppID
            resolved_target = $metadata.target
            raw_arguments = $metadata.arguments
        })
    }
    $successful.Add('start_apps')
} catch {
    $errors['start_apps'] = 'Не удалось получить список приложений Windows'
}

try {
    $folders = @(
        @{ name = 'Рабочий стол'; type = 'DesktopDirectory'; description = 'Файлы и ярлыки рабочего стола'; aliases = @('desktop', 'стол') },
        @{ name = 'Документы'; type = 'MyDocuments'; description = 'Личные документы'; aliases = @('documents', 'мои документы') },
        @{ name = 'Изображения'; type = 'MyPictures'; description = 'Фотографии и изображения'; aliases = @('pictures', 'фото', 'картинки') },
        @{ name = 'Музыка'; type = 'MyMusic'; description = 'Музыка и аудиофайлы'; aliases = @('music', 'аудио') },
        @{ name = 'Видео'; type = 'MyVideos'; description = 'Видео и фильмы'; aliases = @('videos', 'фильмы') },
        @{ name = 'Домашняя папка'; type = 'UserProfile'; description = 'Папка текущего пользователя'; aliases = @('home', 'профиль', 'пользователь') }
    )
    foreach ($folder in $folders) {
        $path = [Environment]::GetFolderPath($folder.type)
        if ($path -and (Test-Path -LiteralPath $path -PathType Container)) {
            $entries.Add([ordered]@{
                name = $folder.name; source = 'known_folders'; source_key = 'known_folders:' + $folder.type
                launch_type = 'folder'; launch_target = $path; resolved_target = $path
                description = $folder.description; aliases = $folder.aliases
            })
        }
    }
    $downloads = $namespaceShell.Namespace('shell:Downloads').Self.Path
    if ($downloads -and (Test-Path -LiteralPath $downloads -PathType Container)) {
        $entries.Add([ordered]@{
            name = 'Загрузки'; source = 'known_folders'; source_key = 'known_folders:Downloads'
            launch_type = 'folder'; launch_target = $downloads; resolved_target = $downloads
            description = 'Скачанные файлы'; aliases = @('downloads', 'скачанное', 'скачанные файлы')
        })
    }
    $successful.Add('known_folders')
} catch {
    $errors['known_folders'] = 'Не удалось получить пользовательские папки Windows'
}

try {
    $projectRoots = @()
    if ($env:JEV_LAUNCHER_FOLDER_ROOTS) {
        $projectRoots = @($env:JEV_LAUNCHER_FOLDER_ROOTS | ConvertFrom-Json)
    }
    foreach ($projectRoot in $projectRoots) {
        # PowerShell 5.1 wraps ConvertFrom-Json's empty array as a null item.
        if (-not $projectRoot) { continue }
        if (-not (Test-Path -LiteralPath $projectRoot -PathType Container)) { continue }
        $directories = @((Get-Item -LiteralPath $projectRoot)) + @(Get-ChildItem -LiteralPath $projectRoot -Directory)
        foreach ($directory in $directories) {
            if ($directory.Name.StartsWith('.') -or $directory.Name -in @('node_modules', '__pycache__', 'venv', 'output', 'build', 'dist', 'tools')) { continue }
            $entries.Add([ordered]@{
                name = $directory.Name; source = 'project_folders'
                source_key = 'project_folders:' + $directory.FullName.ToLowerInvariant()
                launch_type = 'folder'; launch_target = $directory.FullName; resolved_target = $directory.FullName
                description = 'Папка проектов: ' + $directory.Name
                aliases = @($directory.Name)
            })
        }
    }
    $successful.Add('project_folders')
} catch {
    $errors['project_folders'] = 'Не удалось прочитать часть папок проектов'
}

[void][Runtime.InteropServices.Marshal]::ReleaseComObject($shell)
[void][Runtime.InteropServices.Marshal]::ReleaseComObject($namespaceShell)
[ordered]@{
    entries = @($entries.ToArray())
    successful_sources = @($successful.ToArray())
    errors = $errors
} | ConvertTo-Json -Depth 6 -Compress
