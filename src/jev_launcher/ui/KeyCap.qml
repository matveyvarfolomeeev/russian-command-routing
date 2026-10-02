import QtQuick

Rectangle {
    id: control
    property string text: ""
    property bool dark: true
    Theme { id: colors; dark: control.dark }
    implicitWidth: label.implicitWidth + 12
    implicitHeight: 23
    radius: 5
    color: colors.surface
    border.color: colors.line
    Text {
        id: label
        anchors.centerIn: parent
        text: control.text
        color: colors.muted
        font.family: colors.family
        font.pixelSize: 10
    }
}
