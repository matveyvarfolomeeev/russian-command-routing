import QtQuick
import QtQuick.Controls

TextField {
    id: control
    property bool dark: true
    implicitHeight: 40
    color: dark ? "#edf1f7" : "#182536"
    placeholderTextColor: dark ? "#8290a5" : "#748297"
    selectionColor: "#41649a"
    selectedTextColor: "white"
    font.family: "Segoe UI"
    font.pixelSize: 14
    leftPadding: 12
    rightPadding: 12
    background: Rectangle {
        radius: 9
        color: control.dark ? "#151e2c" : "#ffffff"
        border.color: control.activeFocus ? "#83adff" : control.dark ? "#344055" : "#cbd5e3"
        border.width: 1
    }
}
