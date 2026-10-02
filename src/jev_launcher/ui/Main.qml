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
    font.family: colors.family
    property bool dark: controller.theme === "dark"
    property color panel: colors.surface
    property color ink: colors.ink
    property color muted: colors.muted
    property color line: colors.line
    color: colors.background
    Theme { id: colors; dark: window.dark }

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
        spacing: 18
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 38
            spacing: 10
            Rectangle {
                Layout.preferredWidth: 34
                Layout.preferredHeight: 34
                radius: 9
                color: colors.accent
                Text {
                    anchors.centerIn: parent
                    text: "↗"
                    color: colors.accentInk
                    font.family: colors.family
                    font.pixelSize: 24
                    font.weight: Font.DemiBold
                }
            }
            Label {
                text: "Jev"
                color: colors.ink
                font.pixelSize: 20
                font.weight: Font.DemiBold
                font.letterSpacing: -0.5
            }
            Label {
                text: "LAUNCHER"
                color: colors.subtle
                font.pixelSize: 9
                font.letterSpacing: 1.5
                Layout.leftMargin: 2
            }
            Item { Layout.fillWidth: true }
            TabBar {
                id: tabs
                objectName: "navigation"
                Layout.preferredWidth: 286
                Layout.preferredHeight: 38
                spacing: 3
                padding: 3
                background: Rectangle {
                    radius: 9
                    color: colors.input
                    border.color: colors.line
                }
                Repeater {
                    model: ["Поиск", "Каталог", "Настройки"]
                    TabButton {
                        id: tab
                        required property string modelData
                        text: modelData
                        implicitHeight: 32
                        background: Rectangle {
                            radius: 6
                            color: tab.checked ? colors.raised : tab.hovered ? colors.surface : "transparent"
                            border.color: tab.activeFocus ? colors.accent : "transparent"
                        }
                        contentItem: Text {
                            text: tab.text
                            color: tab.checked ? colors.ink : colors.muted
                            font.family: colors.family
                            font.pixelSize: 12
                            font.weight: tab.checked ? Font.DemiBold : Font.Normal
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                    }
                }
            }
        }
        Rectangle { Layout.fillWidth: true; implicitHeight: 1; color: colors.line }
        StackLayout {
            currentIndex: tabs.currentIndex
            Layout.fillWidth: true
            Layout.fillHeight: true

            ColumnLayout {
                spacing: 14
                RowLayout {
                    Layout.fillWidth: true
                    ColumnLayout {
                        spacing: 4
                        Label {
                            text: "Что открыть?"
                            color: colors.ink
                            font.pixelSize: 25
                            font.weight: Font.DemiBold
                            font.letterSpacing: -0.6
                        }
                        Label { text: "Приложения, файлы и папки в одном месте"; color: colors.muted; font.pixelSize: 12 }
                    }
                    Item { Layout.fillWidth: true }
                    Label {
                        text: controller.activeCount + " в каталоге"
                        color: colors.subtle
                        font.pixelSize: 11
                        Layout.alignment: Qt.AlignBottom
                        Layout.bottomMargin: 3
                    }
                }
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 58
                    radius: 11
                    color: colors.surface
                    border.color: searchField.activeFocus ? colors.accent : colors.line
                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 16
                        anchors.rightMargin: 9
                        spacing: 12
                        SearchIcon { ink: colors.muted; Layout.preferredWidth: 22; Layout.preferredHeight: 22 }
                        TextField {
                            id: searchField
                            objectName: "searchField"
                            Layout.fillWidth: true
                            Layout.minimumWidth: 120
                            placeholderText: "Название или описание…"
                            placeholderTextColor: colors.subtle
                            color: colors.ink
                            selectionColor: colors.accent
                            selectedTextColor: colors.accentInk
                            font.pixelSize: 17
                            selectByMouse: true
                            leftPadding: 0
                            rightPadding: 0
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
                            ToolTip.text: "Записать голосовой запрос"
                        }
                        AppButton {
                            visible: controller.recording
                            text: "Отмена"
                            dark: window.dark
                            onClicked: controller.cancelSpeech()
                        }
                    }
                }
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 130
                    radius: 11
                    color: colors.surface
                    border.color: colors.line
                    ListView {
                        id: results
                        objectName: "resultsList"
                        anchors.fill: parent
                        anchors.margins: 7
                        model: controller.results
                        currentIndex: count > 0 ? 0 : -1
                        clip: true
                        spacing: 3
                        boundsBehavior: Flickable.StopAtBounds
                        ScrollBar.vertical: ScrollBar { width: 6 }
                        delegate: Rectangle {
                            id: resultRow
                            required property var modelData
                            required property int index
                            width: results.width - 8
                            height: 66
                            radius: 7
                            color: ListView.isCurrentItem ? colors.selected
                                 : rowMouse.containsMouse ? colors.raised : "transparent"
                            border.color: ListView.isCurrentItem ? colors.selectedBorder : "transparent"
                            MouseArea {
                                id: rowMouse
                                anchors.fill: parent
                                hoverEnabled: true
                                onClicked: { results.currentIndex = resultRow.index; searchField.forceActiveFocus() }
                                onDoubleClicked: controller.openObject(resultRow.modelData.object_id)
                            }
                            RowLayout {
                                anchors.fill: parent
                                anchors.leftMargin: 14
                                anchors.rightMargin: 14
                                spacing: 14
                                Image {
                                    source: resultRow.modelData.icon
                                    asynchronous: false
                                    Layout.preferredWidth: 30
                                    Layout.preferredHeight: 30
                                    sourceSize: Qt.size(30, 30)
                                    fillMode: Image.PreserveAspectFit
                                }
                                ColumnLayout {
                                    Layout.fillWidth: true
                                    Layout.minimumWidth: 0
                                    spacing: 4
                                    Label {
                                        text: resultRow.modelData.name
                                        color: colors.ink
                                        font.pixelSize: 14
                                        font.weight: Font.DemiBold
                                        Layout.fillWidth: true
                                        elide: Text.ElideRight
                                    }
                                    Label {
                                        text: resultRow.modelData.description || resultRow.modelData.kindLabel
                                        color: colors.muted
                                        font.pixelSize: 12
                                        Layout.fillWidth: true
                                        elide: Text.ElideRight
                                    }
                                }
                                Label {
                                    visible: resultRow.modelData.favorite
                                    text: "★"
                                    color: colors.favorite
                                    font.pixelSize: 13
                                }
                                Label { text: resultRow.modelData.kindLabel; color: colors.subtle; font.pixelSize: 10 }
                            }
                            Accessible.role: Accessible.ListItem
                            Accessible.name: modelData.name
                        }
                    }
                    ColumnLayout {
                        anchors.centerIn: parent
                        width: Math.min(360, parent.width - 40)
                        spacing: 10
                        visible: results.count === 0
                        Label {
                            Layout.fillWidth: true
                            horizontalAlignment: Text.AlignHCenter
                            text: controller.scanning ? "Собираем каталог…"
                                : controller.totalCount === 0 ? "Начнём с ваших приложений" : "Совпадений пока нет"
                            color: colors.ink
                            font.pixelSize: 18
                            font.weight: Font.DemiBold
                            wrapMode: Text.Wrap
                        }
                        Label {
                            Layout.fillWidth: true
                            horizontalAlignment: Text.AlignHCenter
                            text: controller.totalCount === 0 ? "Обновите каталог или добавьте файл и папку"
                                  : "Попробуйте другое название или описание"
                            color: colors.muted
                            font.pixelSize: 12
                            wrapMode: Text.Wrap
                        }
                        AppButton {
                            Layout.alignment: Qt.AlignHCenter
                            visible: controller.totalCount === 0
                            text: "Собрать приложения"
                            primary: true
                            dark: window.dark
                            enabled: !controller.scanning
                            onClicked: controller.refresh()
                        }
                    }
                }
                RowLayout {
                    id: selection
                    Layout.fillWidth: true
                    Layout.preferredHeight: 42
                    spacing: 10
                    property var selected: results.currentIndex >= 0 && results.currentIndex < controller.results.length
                                           ? controller.results[results.currentIndex] : null
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.minimumWidth: 0
                        spacing: 4
                        Label {
                            text: selection.selected ? selection.selected.name : "Выберите объект"
                            color: colors.ink
                            font.pixelSize: 12
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }
                        Label {
                            text: selection.selected ? selection.selected.launch_target : controller.hotkeyText + " — вызвать окно"
                            color: colors.subtle
                            font.pixelSize: 10
                            Layout.fillWidth: true
                            elide: Text.ElideMiddle
                        }
                    }
                    AppButton {
                        text: selection.selected && selection.selected.favorite ? "★" : "☆"
                        Layout.preferredWidth: 40
                        dark: window.dark
                        enabled: selection.selected !== null
                        onClicked: controller.toggleFavorite(selection.selected.object_id)
                        ToolTip.visible: hovered
                        ToolTip.text: "Добавить или убрать из избранного"
                    }
                    AppButton {
                        objectName: "openButton"
                        text: "Открыть  ↵"
                        primary: true
                        dark: window.dark
                        enabled: selection.selected !== null && !controller.opening
                        onClicked: controller.openObject(selection.selected.object_id)
                    }
                }
            }
            Catalog { dark: window.dark; ink: colors.ink; muted: colors.muted; panel: colors.surface; line: colors.line }
            Settings { dark: window.dark; ink: colors.ink; muted: colors.muted; panel: colors.surface; line: colors.line }
        }
        Rectangle { Layout.fillWidth: true; implicitHeight: 1; color: colors.line }
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 23
            spacing: 7
            Rectangle {
                Layout.preferredWidth: 5
                Layout.preferredHeight: 5
                radius: 3
                color: controller.recording || controller.isError ? colors.danger : colors.accent
            }
            Label {
                objectName: "statusLabel"
                text: controller.status || controller.catalogState
                color: controller.isError ? colors.danger : colors.muted
                font.pixelSize: 10
                Layout.fillWidth: true
                Layout.minimumWidth: 0
                elide: Text.ElideRight
                ToolTip.visible: statusHover.containsMouse
                ToolTip.text: text
                MouseArea { id: statusHover; anchors.fill: parent; hoverEnabled: true }
            }
            KeyCap { text: "↑ ↓"; dark: window.dark; visible: tabs.currentIndex === 0 }
            Label { text: "выбор"; color: colors.subtle; font.pixelSize: 10; visible: tabs.currentIndex === 0 }
            KeyCap { text: "Enter"; dark: window.dark; visible: tabs.currentIndex === 0; Layout.leftMargin: 6 }
            Label { text: "открыть"; color: colors.subtle; font.pixelSize: 10; visible: tabs.currentIndex === 0 }
            KeyCap { text: "Esc"; dark: window.dark; Layout.leftMargin: 6 }
            Label { text: "скрыть"; color: colors.subtle; font.pixelSize: 10 }
        }
    }
    Component.onCompleted: searchField.forceActiveFocus()
}
