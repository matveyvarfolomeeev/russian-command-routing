import QtQuick
import QtQuick.Controls

ComboBox {
    id: control
    property bool dark: true
    Theme { id: colors; dark: control.dark }
    implicitHeight: 42
    implicitWidth: 180
    leftPadding: 12
    rightPadding: 34
    font.family: colors.family
    font.pixelSize: 13
    contentItem: Text {
        text: control.displayText
        font: control.font
        color: colors.ink
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
    indicator: Text {
        x: control.width - 26
        y: (control.height - height) / 2
        text: "⌄"
        color: colors.muted
        font.pixelSize: 20
    }
    background: Rectangle {
        radius: 8
        color: colors.input
        border.color: control.activeFocus ? colors.accent : colors.line
    }
    delegate: ItemDelegate {
        width: control.width - 2
        height: 38
        highlighted: control.highlightedIndex === index
        contentItem: Text {
            text: control.textRole ? modelData[control.textRole] : modelData
            font: control.font
            color: colors.ink
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }
        background: Rectangle {
            radius: 6
            color: parent.highlighted ? colors.selected : "transparent"
        }
    }
    popup: Popup {
        y: control.height + 4
        width: control.width
        padding: 5
        implicitHeight: Math.min(contentItem.implicitHeight + 10, 230)
        contentItem: ListView {
            clip: true
            implicitHeight: contentHeight
            model: control.popup.visible ? control.delegateModel : null
            currentIndex: control.highlightedIndex
            boundsBehavior: Flickable.StopAtBounds
            ScrollIndicator.vertical: ScrollIndicator {}
        }
        background: Rectangle {
            radius: 9
            color: colors.surface
            border.color: colors.line
        }
    }
}
