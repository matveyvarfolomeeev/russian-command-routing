import QtQuick
import QtQuick.Controls

CheckBox {
    id: control
    property bool dark: true
    Theme { id: colors; dark: control.dark }
    implicitHeight: 36
    spacing: 9
    padding: 0
    hoverEnabled: true
    indicator: Rectangle {
        implicitWidth: 19
        implicitHeight: 19
        x: control.leftPadding
        y: (control.height - height) / 2
        radius: 5
        color: control.checked ? colors.accent : colors.input
        border.color: control.activeFocus || control.hovered ? colors.accent : colors.line
        Text {
            anchors.centerIn: parent
            text: control.checked ? "✓" : ""
            color: colors.accentInk
            font.pixelSize: 14
            font.weight: Font.DemiBold
        }
    }
    contentItem: Text {
        text: control.text
        color: colors.ink
        font.family: colors.family
        font.pixelSize: 13
        leftPadding: control.indicator.width + control.spacing
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
}
