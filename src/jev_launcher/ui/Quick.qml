import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects

Window {
    id: window
    objectName: "quickWindow"
    title: "Jev Launcher Quick"
    flags: Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
    color: "transparent"
    visible: quick.visible
    property bool dark: controller.theme === "dark"
    property real centerX: 600
    property real centerY: 400
    property int availableWidth: 1920
    property int availableHeight: 1080
    property bool compact: quick.submitted
    property bool reduceMotion: !controller.animationsEnabled
    width: compact ? 220 : Math.min(680, availableWidth - 24)
    height: compact ? 84 : Math.min(quick.error ? 250 : 218, availableHeight - 24)
    x: Math.round(centerX - width / 2)
    y: Math.round(centerY - height / 2)
    Behavior on width { NumberAnimation { duration: window.visible && !window.reduceMotion ? 180 : 0; easing.type: Easing.OutCubic } }
    Behavior on height { NumberAnimation { duration: window.visible && !window.reduceMotion ? 180 : 0; easing.type: Easing.OutCubic } }
    Theme { id: colors; dark: window.dark }

    function focusInput() {
        if (quick.recording || quick.submitted)
            keyboard.forceActiveFocus()
        else
            input.forceActiveFocus()
    }
    // Native activation can emit a transient inactive event while Windows changes focus.
    onActiveChanged: { if (visible && !active) focusCheck.restart(); else focusCheck.stop() }
    Timer {
        id: focusCheck
        interval: 0
        onTriggered: { if (window.visible && !window.active) quick.focusLost() }
    }
    onClosing: function(close) { close.accepted = false; controller.cancelQuick() }
    onVisibleChanged: {
        if (visible) {
            surface.opacity = 0
            appear.restart()
            focusInput()
        }
    }
    Shortcut { sequence: "Escape"; context: Qt.WindowShortcut; onActivated: controller.cancelQuick() }
    Connections {
        target: quick
        function onChanged() {
            if (input.text !== quick.query) input.text = quick.query
            if (window.active) window.focusInput()
        }
    }
    NumberAnimation { id: appear; target: surface; property: "opacity"; to: 1; duration: window.reduceMotion ? 0 : 160; easing.type: Easing.OutCubic }

    Item {
        id: keyboard
        anchors.fill: parent
        focus: true
        Keys.onReturnPressed: function(event) { controller.submitQuick(); event.accepted = true }
        Keys.onEnterPressed: function(event) { controller.submitQuick(); event.accepted = true }
        Rectangle {
            id: surface
            objectName: "quickSurface"
            anchors.fill: parent
            anchors.margins: 20
            radius: window.compact ? 22 : 18
            color: window.dark ? "#f2262b33" : "#f7ffffff"
            border.color: window.dark ? "#515a69" : "#d4dae4"
            layer.enabled: true
            layer.effect: MultiEffect {
                shadowEnabled: true
                shadowBlur: 0.7
                shadowColor: "#68000000"
                shadowVerticalOffset: 5
            }
            RowLayout {
                visible: window.compact
                anchors.fill: parent
                anchors.leftMargin: 14
                anchors.rightMargin: 7
                spacing: 9
                Item {
                    Layout.preferredWidth: 18
                    Layout.preferredHeight: 18
                    Rectangle { anchors.fill: parent; radius: 9; color: "transparent"; border.color: colors.line; border.width: 2 }
                    Rectangle { width: 5; height: 5; radius: 3; color: colors.accent; x: 6.5; y: 0 }
                    RotationAnimator on rotation {
                        from: 0; to: 360; duration: 850; loops: Animation.Infinite
                        running: window.compact && !window.reduceMotion
                    }
                }
                Label {
                    objectName: "quickStageLabel"
                    Layout.fillWidth: true
                    color: colors.ink
                    font.family: colors.family
                    font.pixelSize: 13
                    text: quick.stage === "transcribing" ? "Распознаю" : quick.stage === "opening" ? "Открываю" : "Ищу"
                }
                AppButton {
                    objectName: "quickCancelButton"
                    text: "×"
                    compact: true
                    dark: window.dark
                    implicitWidth: 27
                    implicitHeight: 28
                    padding: 0
                    leftPadding: 0
                    rightPadding: 0
                    topPadding: 0
                    bottomPadding: 0
                    Accessible.name: "Отменить запуск"
                    onClicked: controller.cancelQuick()
                }
            }
            ColumnLayout {
                visible: !window.compact
                anchors.fill: parent
                anchors.margins: 18
                spacing: 12
                RowLayout {
                    visible: !quick.recording
                    Layout.fillWidth: true
                    spacing: 12
                    SearchIcon { ink: colors.muted; Layout.preferredWidth: 22; Layout.preferredHeight: 22 }
                    TextField {
                        id: input
                        objectName: "quickInput"
                        Layout.fillWidth: true
                        Layout.minimumWidth: 0
                        implicitHeight: 34
                        placeholderText: "Что открыть?"
                        placeholderTextColor: colors.subtle
                        color: colors.ink
                        selectionColor: colors.accent
                        selectedTextColor: colors.accentInk
                        font.family: colors.family
                        font.pixelSize: 20
                        background: null
                        padding: 0
                        selectByMouse: true
                        onTextEdited: quick.setQuery(text)
                        onAccepted: controller.submitQuick()
                        Accessible.name: "Запрос для быстрого запуска"
                    }
                    AppButton {
                        objectName: "quickMicButton"
                        text: "\uE720"
                        font.family: "Segoe MDL2 Assets"
                        font.pixelSize: 18
                        dark: window.dark
                        compact: true
                        enabled: quick.stage !== "transcribing"
                        Accessible.name: "Включить микрофон"
                        onClicked: quick.startRecording()
                        ToolTip.visible: hovered
                        ToolTip.text: "Записать голосовой запрос"
                    }
                }
                RowLayout {
                    visible: quick.recording
                    Layout.fillWidth: true
                    spacing: 12
                    Rectangle {
                        width: 9; height: 9; radius: 5; color: colors.danger
                        SequentialAnimation on opacity {
                            running: quick.recording && !window.reduceMotion
                            loops: Animation.Infinite
                            NumberAnimation { to: 0.35; duration: 650 }
                            NumberAnimation { to: 1; duration: 650 }
                        }
                    }
                    Label { text: "Слушаю…"; color: colors.ink; font.pixelSize: 20; Layout.fillWidth: true }
                    AppButton { objectName: "quickStopButton"; text: "Стоп"; dark: window.dark; onClicked: quick.stopRecording() }
                    AppButton { text: "Отмена"; dark: window.dark; onClicked: controller.cancelQuick() }
                }
                Rectangle { Layout.fillWidth: true; implicitHeight: 1; color: colors.line }
                Rectangle {
                    objectName: "quickResultCard"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 42
                    radius: 10
                    color: quick.bestResult.object_id ? colors.selected : "transparent"
                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 12
                        Image {
                            visible: !!quick.bestResult.object_id
                            source: quick.bestResult.icon || ""
                            Layout.preferredWidth: 28
                            Layout.preferredHeight: 28
                            sourceSize: Qt.size(28, 28)
                            fillMode: Image.PreserveAspectFit
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            Layout.minimumWidth: 0
                            spacing: 3
                            Label {
                                text: quick.recording ? "Скажите название или описание"
                                    : quick.stage === "transcribing" ? "Распознаю речь…"
                                    : quick.bestResult.name || (quick.stage === "searching" ? "Ищу подходящий объект…" : quick.query ? "Совпадений пока нет" : "Приложения, файлы и папки")
                                color: colors.ink
                                font.pixelSize: 14
                                font.weight: Font.DemiBold
                                Layout.fillWidth: true
                                elide: Text.ElideRight
                            }
                            Label {
                                visible: !!quick.bestResult.object_id
                                text: quick.bestResult.kindLabel || ""
                                color: colors.muted
                                font.pixelSize: 11
                            }
                        }
                    }
                }
                Label {
                    objectName: "quickErrorLabel"
                    visible: !!quick.error
                    Layout.fillWidth: true
                    text: quick.error
                    color: colors.danger
                    font.pixelSize: 12
                    wrapMode: Text.Wrap
                    maximumLineCount: 3
                    elide: Text.ElideRight
                    ToolTip.visible: errorHover.containsMouse
                    ToolTip.text: text
                    MouseArea { id: errorHover; anchors.fill: parent; hoverEnabled: true }
                }
                RowLayout {
                    Layout.fillWidth: true
                    Label { text: "Enter — открыть"; color: colors.muted; font.pixelSize: 11 }
                    Item { Layout.fillWidth: true }
                    Label { text: "Esc — отмена"; color: colors.subtle; font.pixelSize: 11 }
                }
            }
        }
    }
}
