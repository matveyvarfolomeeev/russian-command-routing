import QtQuick
import QtQuick.Controls

Button {
    id: control
    property bool primary: false
    property bool compact: false
    property bool dark: true
    property bool destructive: false
    property color accent: colors.accent
    Theme { id: colors; dark: control.dark }
    implicitHeight: compact ? 34 : 40
    implicitWidth: Math.max(compact ? 36 : 72, contentItem.implicitWidth + 28)
    leftPadding: 14
    rightPadding: 14
    topPadding: 7
    bottomPadding: 7
    hoverEnabled: true
    font.family: colors.family
    font.pixelSize: 13
    font.weight: primary ? Font.DemiBold : Font.Normal
    contentItem: Text {
        text: control.text
        font: control.font
        color: !control.enabled ? colors.subtle : control.primary ? colors.accentInk
              : control.destructive ? colors.danger : colors.ink
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
    background: Rectangle {
        radius: 8
        color: control.primary ? (control.down || control.hovered ? colors.accentHover : control.accent)
             : control.down ? colors.line : control.hovered ? colors.raised : colors.surface
        border.width: 1
        border.color: control.activeFocus ? colors.accent : control.primary ? "transparent" : colors.line
        opacity: control.enabled ? 1 : 0.55
    }
}
