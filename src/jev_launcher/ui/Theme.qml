import QtQuick

QtObject {
    property bool dark: true
    readonly property color background: dark ? "#191d24" : "#f3f5f9"
    readonly property color surface: dark ? "#242a33" : "#ffffff"
    readonly property color raised: dark ? "#303844" : "#edf1f7"
    readonly property color input: dark ? "#20252d" : "#f8faff"
    readonly property color ink: dark ? "#edf2fa" : "#202c3d"
    readonly property color muted: dark ? "#a8b3c4" : "#647287"
    readonly property color subtle: dark ? "#8793a6" : "#78869a"
    readonly property color line: dark ? "#3a4554" : "#dbe2ec"
    readonly property color accent: dark ? "#94bcf5" : "#356cc0"
    readonly property color accentHover: dark ? "#b0cffe" : "#2859a1"
    readonly property color accentInk: dark ? "#172840" : "#ffffff"
    readonly property color selected: dark ? "#2e3d54" : "#eaf1fd"
    readonly property color selectedBorder: dark ? "#506b94" : "#bfd2f2"
    readonly property color danger: dark ? "#f1aaa0" : "#a13e32"
    readonly property color favorite: dark ? "#dcca95" : "#8b7338"
    readonly property string family: "Segoe UI Variable"
}
