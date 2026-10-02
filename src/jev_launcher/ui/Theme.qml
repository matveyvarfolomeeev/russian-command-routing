import QtQuick

QtObject {
    property bool dark: true
    readonly property color background: dark ? "#171a18" : "#f4f5f1"
    readonly property color surface: dark ? "#202420" : "#ffffff"
    readonly property color raised: dark ? "#2a302a" : "#ecefe8"
    readonly property color input: dark ? "#1b1f1c" : "#fafbf8"
    readonly property color ink: dark ? "#edf0e9" : "#222a23"
    readonly property color muted: dark ? "#a1aaa0" : "#697267"
    readonly property color subtle: dark ? "#777f76" : "#838c81"
    readonly property color line: dark ? "#353b34" : "#dce2d7"
    readonly property color accent: dark ? "#bddb9d" : "#456636"
    readonly property color accentHover: dark ? "#cdeab0" : "#355529"
    readonly property color accentInk: dark ? "#21301b" : "#ffffff"
    readonly property color selected: dark ? "#2d3829" : "#edf3e6"
    readonly property color selectedBorder: dark ? "#4a5b40" : "#c9dabc"
    readonly property color danger: dark ? "#f1aaa0" : "#a13e32"
    readonly property color favorite: dark ? "#dcca95" : "#8b7338"
    readonly property string family: "Segoe UI Variable"
}
