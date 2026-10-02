import QtQuick
import QtQuick.Controls

Button {
    id: control
    property bool primary: false
    property bool compact: false
    property bool dark: true
    property color accent: "#83adff"
    implicitHeight: compact ? 32 : 40
    implicitWidth: Math.max(compact ? 32 : 72, contentItem.implicitWidth + 24)
    padding: 12
    hoverEnabled: true
    font.family: "Segoe UI"
    font.pixelSize: 13
    contentItem: Text {
        text: control.text
        font: control.font
        color: !control.enabled ? "#78818f" : control.primary ? "#15213a"
              : control.dark ? "#dce3ed" : "#26354a"
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }
    background: Rectangle {
        radius: 9
        color: control.primary ? (control.down ? "#6b98ec" : control.accent)
             : control.down ? (control.dark ? "#344158" : "#dce5f3")
             : control.hovered ? (control.dark ? "#2a3548" : "#e7edf6")
             : (control.dark ? "#202938" : "#f0f3f8")
        border.width: control.activeFocus ? 2 : 1
        border.color: control.activeFocus ? control.accent
                    : control.primary ? "transparent" : control.dark ? "#313c4e" : "#dce3ee"
        opacity: control.enabled ? 1 : 0.6
    }
}
