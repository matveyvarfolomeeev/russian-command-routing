import QtQuick

Canvas {
    id: icon
    property color ink: "#a1aaa0"
    implicitWidth: 22
    implicitHeight: 22
    onInkChanged: requestPaint()
    onPaint: {
        const context = getContext("2d")
        context.reset()
        context.strokeStyle = ink
        context.lineWidth = 1.7
        context.lineCap = "round"
        context.beginPath()
        context.arc(9, 9, 6, 0, Math.PI * 2)
        context.moveTo(13.5, 13.5)
        context.lineTo(19, 19)
        context.stroke()
    }
}
