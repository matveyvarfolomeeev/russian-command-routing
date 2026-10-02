import QtQuick
import QtQuick.Controls

TextField {
    id: control
    property bool dark: true
    Theme { id: colors; dark: control.dark }
    implicitHeight: 42
    color: colors.ink
    placeholderTextColor: colors.subtle
    selectionColor: colors.accent
    selectedTextColor: colors.accentInk
    font.family: colors.family
    font.pixelSize: 13
    selectByMouse: true
    leftPadding: 12
    rightPadding: 12
    topPadding: 9
    bottomPadding: 9
    background: Rectangle {
        radius: 8
        color: colors.input
        border.color: control.activeFocus ? colors.accent : colors.line
        border.width: 1
    }
}
