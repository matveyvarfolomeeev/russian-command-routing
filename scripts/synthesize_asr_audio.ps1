param([Parameter(Mandatory = $true)][string]$Plan)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$items = Get-Content -LiteralPath $Plan -Raw -Encoding UTF8 | ConvertFrom-Json
$format = [System.Speech.AudioFormat.SpeechAudioFormatInfo]::new(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono
)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $available = @($synth.GetInstalledVoices() | Where-Object { $_.Enabled } |
        ForEach-Object { $_.VoiceInfo.Name })
    $required = @($items | ForEach-Object { $_.voice } | Sort-Object -Unique)
    foreach ($name in $required) {
        if ($name -notin $available) { throw "Required TTS voice unavailable: $name" }
    }
    $done = 0
    foreach ($item in $items) {
        if (-not (Test-Path -LiteralPath $item.output)) {
            $parent = Split-Path -Path $item.output -Parent
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
            $synth.SelectVoice($item.voice)
            $synth.Rate = [int]$item.rate
            $synth.Volume = 100
            $synth.SetOutputToWaveFile($item.output, $format)
            $synth.Speak([string]$item.text)
            $synth.SetOutputToNull()
        }
        $done++
        if ($done % 25 -eq 0 -or $done -eq $items.Count) {
            Write-Output ('{"tts_completed":' + $done + ',"total":' + $items.Count + '}')
        }
    }
} finally {
    $synth.Dispose()
}
