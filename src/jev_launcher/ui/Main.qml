import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window
    objectName: "launcherWindow"
    width: 840
    height: 670
    minimumWidth: 720
    minimumHeight: 560
    visible: true
    title: "Jev Launcher"
    flags: Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowMinMaxButtonsHint
    font.family: "Segoe UI"
    property bool dark: controller.theme === "dark"
    property color panel: dark ? "#192230" : "#ffffff"
    property color ink: dark ? "#edf1f7" : "#19283d"
    property color muted: dark ? "#8d9bb0" : "#6b7b90"
    property color line: dark ? "#2b3648" : "#dce4ef"
    color: dark ? "#111925" : "#f3f6fb"

    function focusSearch() {
        tabs.currentIndex = 0
        searchField.forceActiveFocus()
        searchField.selectAll()
    }
    onClosing: function(close) {
        close.accepted = false
        controller.hideWindow()
    }
    Shortcut { sequence: "Escape"; onActivated: controller.hideWindow() }
    Shortcut { sequence: "Ctrl+L"; onActivated: window.focusSearch() }
    Connections {
        target: controller
        function onQueryChanged() {
            searchField.text = controller.query
            results.currentIndex = 0
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 24
        spacing: 16
        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            Image { source: "../resources/launcher.svg"; sourceSize.width: 38; sourceSize.height: 38 }
            ColumnLayout {
                spacing: 1
                Label { text: "Jev Launcher"; color: window.ink; font.pixelSize: 17; font.weight: Font.DemiBold }
                Label { text: "Приложения и ваши пространства"; color: window.muted; font.pixelSize: 11 }
            }
            Item { Layout.fillWidth: true }
            TabBar {
                id: tabs
                objectName: "navigation"
                Layout.preferredWidth: 310
                background: Rectangle { radius: 10; color: window.panel; border.color: window.line }
                Repeater {
                    model: ["Поиск", "Каталог", "Настройки"]
                    TabButton {
                        required property string modelData
                        text: modelData
                        implicitHeight: 38
                        background: Rectangle {
                            radius: 8
                            color: parent.checked ? (window.dark ? "#2b3c58" : "#e3ecfd") : "transparent"
                        }
                        contentItem: Text {
                            text: parent.text
                            color: parent.checked ? (window.dark ? "#b8d0ff" : "#345989") : window.muted
                            font.pixelSize: 13
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                    }
                }
            }
        }
        Rectangle { Layout.fillWidth: true; height: 1; color: window.line }
        StackLayout {
            currentIndex: tabs.currentIndex
            Layout.fillWidth: true
            Layout.fillHeight: true

            ColumnLayout {
                spacing: 14
                RowLayout {
                    ColumnLayout {
                        spacing: 4
                        Label { text: "Что открыть?"; color: window.ink; font.pixelSize: 28; font.weight: Font.DemiBold }
                        Label { text: "Найдите объект и нажмите Enter"; color: window.muted; font.pixelSize: 13 }
                    }
                    Item { Layout.fillWidth: true }
                    Label { text: controller.activeCount + " объектов"; color: window.muted; font.pixelSize: 12 }
                }
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: 60
                    radius: 12
                    color: window.panel
                    border.color: searchField.activeFocus ? "#7099e5" : window.line
                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 10
                        Label { text: "⌕"; color: window.muted; font.pixelSize: 26; Layout.leftMargin: 5 }
                        TextField {
                            id: searchField
                            objectName: "searchField"
                            Layout.fillWidth: true
                            placeholderText: "Название, папка или описание…"
                            placeholderTextColor: window.muted
                            color: window.ink
                            font.pixelSize: 17
                            selectByMouse: true
                            background: null
                            onTextEdited: { controller.setQuery(text); results.currentIndex = 0 }
                            onAccepted: {
                                if (results.count > 0)
                                    controller.openObject(controller.results[Math.max(0, results.currentIndex)].object_id)
                            }
                            Keys.onDownPressed: function(event) {
                                results.currentIndex = Math.min(results.count - 1, results.currentIndex + 1)
                                event.accepted = true
                            }
                            Keys.onUpPressed: function(event) {
                                results.currentIndex = Math.max(0, results.currentIndex - 1)
                                event.accepted = true
                            }
                        }
                        AppButton {
                            objectName: "voiceButton"
                            text: controller.recording ? "■ Стоп" : controller.transcribing ? "Отмена" : "Голос"
                            primary: controller.recording
                            dark: window.dark
                            onClicked: controller.toggleRecording()
                            ToolTip.visible: hovered
                            ToolTip.text: "Запись до 60 секунд, распознавание через Groq"
                        }
                        AppButton {
                            visible: controller.recording
                            text: "Отмена"; dark: window.dark
                            onClicked: controller.cancelSpeech()
                        }
                    }
                }
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: 12
                    color: window.panel
                    border.color: window.line
                    ListView {
                        id: results
                        objectName: "resultsList"
                        anchors.fill: parent
                        anchors.margins: 8
                        model: controller.results
                        currentIndex: count > 0 ? 0 : -1
                        clip: true
                        spacing: 3
                        boundsBehavior: Flickable.StopAtBounds
                        ScrollBar.vertical: ScrollBar {}
                        delegate: Rectangle {
                            id: resultRow
                            required property var modelData
                            required property int index
                            width: results.width
                            height: 62
                            radius: 9
                            color: ListView.isCurrentItem ? (window.dark ? "#253650" : "#e7effd")
                                 : rowMouse.containsMouse ? (window.dark ? "#202d40" : "#f2f5fa") : "transparent"
                            MouseArea {
                                id: rowMouse
                                anchors.fill: parent
                                hoverEnabled: true
                                onClicked: { results.currentIndex = resultRow.index; searchField.forceActiveFocus() }
                                onDoubleClicked: controller.openObject(resultRow.modelData.object_id)
                            }
                            RowLayout {
                                anchors.fill: parent
                                anchors.leftMargin: 12; anchors.rightMargin: 12
                                spacing: 12
                                Image {
                                    source: resultRow.modelData.icon
                                    asynchronous: false
                                    Layout.preferredWidth: 32; Layout.preferredHeight: 32
                                    sourceSize: Qt.size(32, 32)
                                    fillMode: Image.PreserveAspectFit
                                }
                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 3
                                    Label {
                                        text: resultRow.modelData.name
                                        color: window.ink; font.pixelSize: 14; font.weight: Font.DemiBold
                                        Layout.fillWidth: true; elide: Text.ElideRight
                                    }
                                    Label {
                                        text: resultRow.modelData.description || resultRow.modelData.kindLabel
                                        color: window.muted; font.pixelSize: 12
                                        Layout.fillWidth: true; elide: Text.ElideRight
                                    }
                                }
                                Label { text: resultRow.modelData.favorite ? "★" : ""; color: "#d7bb74"; font.pixelSize: 15 }
                                Label { text: resultRow.modelData.kindLabel; color: window.muted; font.pixelSize: 11 }
                            }
                            Accessible.role: Accessible.ListItem
                            Accessible.name: modelData.name
                        }
                    }
                    Column {
                        anchors.centerIn: parent
                        spacing: 8
                        visible: results.count === 0
                        Label {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: controller.scanning ? "Собираем каталог…"
                                : controller.totalCount === 0 ? "Начнём с ваших приложений" : "Совпадений не найдено"
                            color: window.ink; font.pixelSize: 18
                        }
                        Label {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: controller.totalCount === 0 ? "Обновите каталог или добавьте объект вручную"
                                  : "Попробуйте другое название или описание"
                            color: window.muted; font.pixelSize: 13
                        }
                        AppButton {
                            anchors.horizontalCenter: parent.horizontalCenter
                            visible: controller.totalCount === 0
                            text: "Собрать приложения"; primary: true; enabled: !controller.scanning
                            onClicked: controller.refresh()
                        }
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    property var selected: results.currentIndex >= 0 && results.currentIndex < controller.results.length
                                           ? controller.results[results.currentIndex] : null
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 3
                        Label {
                            text: parent.parent.selected ? parent.parent.selected.name : "Готов к поиску"
                            color: window.ink; font.pixelSize: 13
                            Layout.fillWidth: true; elide: Text.ElideRight
                        }
                        Label {
                            text: parent.parent.selected ? parent.parent.selected.launch_target : controller.hotkeyText + " — вызвать окно"
                            color: window.muted; font.pixelSize: 11
                            Layout.fillWidth: true; elide: Text.ElideMiddle
                        }
                    }
                    AppButton {
                        text: "☆"; dark: window.dark
                        enabled: parent.selected !== null
                        onClicked: controller.toggleFavorite(parent.selected.object_id)
                        ToolTip.visible: hovered; ToolTip.text: "Добавить или убрать из избранного"
                    }
                    AppButton {
                        objectName: "openButton"
                        text: "Открыть ↵"; primary: true
                        enabled: parent.selected !== null && !controller.opening
                        onClicked: controller.openObject(parent.selected.object_id)
                    }
                }
            }
            Catalog { dark: window.dark; ink: window.ink; muted: window.muted; panel: window.panel; line: window.line }
            Settings { dark: window.dark; ink: window.ink; muted: window.muted; panel: window.panel; line: window.line }
        }
        Rectangle { Layout.fillWidth: true; height: 1; color: window.line }
        RowLayout {
            Layout.fillWidth: true
            Label {
                text: "●"
                color: controller.recording || controller.isError ? "#f08b8b" : "#8cc9ae"
                font.pixelSize: 11
            }
            Label {
                objectName: "statusLabel"
                text: controller.status || "Готово"
                color: controller.isError ? (window.dark ? "#f0a3a3" : "#a73e3e") : window.muted
                font.pixelSize: 11
                Layout.fillWidth: true
                elide: Text.ElideRight
                ToolTip.visible: statusHover.containsMouse
                ToolTip.text: text
                MouseArea { id: statusHover; anchors.fill: parent; hoverEnabled: true }
            }
            Label { text: "↑ ↓ выбор   ·   Enter открыть"; color: window.muted; font.pixelSize: 11 }
        }
    }
    Component.onCompleted: searchField.forceActiveFocus()
}
