import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ColumnLayout {
    id: page
    objectName: "catalogPage"
    property bool dark: true
    property color ink
    property color muted
    property color panel
    property color line
    Theme { id: colors; dark: page.dark }
    spacing: 14

    function editObject(item) {
        editor.record = item
        nameField.text = item.name
        descriptionField.text = item.description
        aliasesField.text = item.aliases.join(", ")
        enabledBox.checked = item.enabled
        favoriteBox.checked = item.favorite
        editError.text = ""
        editor.open()
    }

    ColumnLayout {
        Layout.fillWidth: true
        spacing: 8
        ColumnLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            spacing: 4
            Label {
                text: "Ваш каталог"
                color: colors.ink
                font.pixelSize: 25
                font.weight: Font.DemiBold
                font.letterSpacing: -0.6
                Layout.fillWidth: true
                elide: Text.ElideRight
            }
            Label {
                text: controller.activeCount + " доступно · " + controller.totalCount + " всего"
                color: colors.muted
                font.pixelSize: 12
                Layout.fillWidth: true
                elide: Text.ElideRight
            }
        }
        RowLayout {
        Layout.fillWidth: true
        spacing: 8
        Item { Layout.fillWidth: true }
        AppButton { text: "+ Файл"; dark: page.dark; onClicked: controller.addPath(false) }
        AppButton { text: "+ Папка"; dark: page.dark; onClicked: controller.addPath(true) }
        AppButton {
            objectName: "refreshButton"
            text: controller.scanning ? "Обновляем…" : "Обновить"
            dark: page.dark
            enabled: !controller.scanning
            onClicked: controller.refresh()
        }
        }
    }
    Field {
        objectName: "catalogSearchField"
        Layout.fillWidth: true
        dark: page.dark
        placeholderText: "Поиск по каталогу, включая скрытые записи"
        onTextEdited: controller.setCatalogQuery(text)
    }
    Rectangle {
        Layout.fillWidth: true
        Layout.fillHeight: true
        Layout.minimumHeight: 130
        color: colors.surface
        radius: 11
        border.color: colors.line
        ListView {
            id: catalogList
            objectName: "catalogList"
            anchors.fill: parent
            anchors.margins: 7
            clip: true
            spacing: 3
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar { width: 6 }
            model: controller.catalogItems
            delegate: Rectangle {
                id: row
                required property var modelData
                width: catalogList.width - 8
                height: 66
                radius: 7
                color: rowMouse.containsMouse ? colors.raised : "transparent"
                MouseArea {
                    id: rowMouse
                    anchors.fill: parent
                    hoverEnabled: true
                    onDoubleClicked: page.editObject(row.modelData)
                }
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 14
                    anchors.rightMargin: 10
                    spacing: 14
                    Image {
                        source: row.modelData.icon
                        asynchronous: false
                        Layout.preferredWidth: 30
                        Layout.preferredHeight: 30
                        sourceSize: Qt.size(30, 30)
                        fillMode: Image.PreserveAspectFit
                        opacity: row.modelData.enabled && row.modelData.available ? 1 : 0.45
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.minimumWidth: 0
                        spacing: 4
                        Label {
                            text: row.modelData.name
                            color: row.modelData.enabled && row.modelData.available ? colors.ink : colors.muted
                            font.pixelSize: 14
                            font.weight: Font.DemiBold
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }
                        Label {
                            text: !row.modelData.available ? "Недоступен · " + row.modelData.sourceLabel
                                 : !row.modelData.enabled ? "Скрыт из поиска · " + row.modelData.sourceLabel
                                 : row.modelData.description || row.modelData.kindLabel
                            color: colors.muted
                            font.pixelSize: 12
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }
                    }
                    Label { visible: row.modelData.favorite; text: "★"; color: colors.favorite; font.pixelSize: 13 }
                    AppButton {
                        text: "Править"
                        compact: true
                        dark: page.dark
                        onClicked: page.editObject(row.modelData)
                    }
                }
            }
        }
        Label {
            anchors.centerIn: parent
            width: parent.width - 40
            visible: catalogList.count === 0
            text: "Здесь появятся ваши приложения и папки"
            horizontalAlignment: Text.AlignHCenter
            wrapMode: Text.Wrap
            color: colors.muted
            font.pixelSize: 14
        }
    }
    RowLayout {
        Layout.fillWidth: true
        spacing: 8
        ColumnLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            spacing: 3
            Label {
                text: controller.scanning ? "Обновляется в фоне" : "Каталог сохранён на этом компьютере"
                color: colors.muted
                font.pixelSize: 11
                Layout.fillWidth: true
                elide: Text.ElideRight
            }
            Label {
                text: "Обновление: " + controller.lastScan
                color: colors.subtle
                font.pixelSize: 10
                Layout.fillWidth: true
                elide: Text.ElideRight
            }
        }
        AppButton {
            text: "Экспорт каталога"
            compact: true
            dark: page.dark
            enabled: controller.activeCount > 0
            onClicked: controller.exportCatalog()
        }
    }
    Dialog {
        id: editor
        objectName: "objectEditor"
        parent: Overlay.overlay
        width: Math.min(570, parent.width - 40)
        height: Math.min(570, parent.height - 40)
        x: (parent.width - width) / 2
        y: (parent.height - height) / 2
        modal: true
        property var record: ({})
        padding: 24
        closePolicy: Popup.CloseOnEscape
        background: Rectangle { color: colors.surface; radius: 14; border.color: colors.line }
        Overlay.modal: Rectangle { color: page.dark ? "#b30a0e0b" : "#66212920" }
        header: Item {
            implicitHeight: 70
            ColumnLayout {
                anchors.fill: parent
                anchors.leftMargin: 24
                anchors.rightMargin: 24
                anchors.topMargin: 22
                anchors.bottomMargin: 4
                spacing: 3
                Label {
                    text: "Описание объекта"
                    color: colors.ink
                    font.pixelSize: 21
                    font.weight: Font.DemiBold
                    font.letterSpacing: -0.4
                }
                Label { text: "Сделайте его проще найти"; color: colors.muted; font.pixelSize: 12 }
            }
        }
        contentItem: ScrollView {
            id: editorScroll
            clip: true
            contentWidth: availableWidth
            ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
            ColumnLayout {
                width: editorScroll.availableWidth
                spacing: 9
                Label { text: "Название"; color: colors.muted; font.pixelSize: 12 }
                Field { id: nameField; objectName: "nameField"; dark: page.dark; Layout.fillWidth: true }
                Label { text: "Для чего этот объект"; color: colors.muted; font.pixelSize: 12; Layout.topMargin: 4 }
                TextArea {
                    id: descriptionField
                    objectName: "descriptionField"
                    Layout.fillWidth: true
                    Layout.preferredHeight: 82
                    wrapMode: TextEdit.Wrap
                    color: colors.ink
                    placeholderText: "Например: проект по машинному обучению"
                    placeholderTextColor: colors.subtle
                    selectionColor: colors.accent
                    selectedTextColor: colors.accentInk
                    font.pixelSize: 13
                    padding: 12
                    background: Rectangle {
                        radius: 8
                        color: colors.input
                        border.color: descriptionField.activeFocus ? colors.accent : colors.line
                    }
                }
                Label { text: "Другие названия через запятую"; color: colors.muted; font.pixelSize: 12; Layout.topMargin: 4 }
                Field { id: aliasesField; objectName: "aliasesField"; dark: page.dark; Layout.fillWidth: true }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 22
                    AppCheck { id: enabledBox; text: "Показывать в поиске"; dark: page.dark }
                    AppCheck { id: favoriteBox; text: "Избранное"; dark: page.dark }
                    Item { Layout.fillWidth: true }
                }
                Rectangle { Layout.fillWidth: true; implicitHeight: 1; color: colors.line; Layout.topMargin: 2 }
                Label {
                    text: (editor.record.kindLabel || "") + " · " + (editor.record.sourceLabel || "")
                    color: colors.subtle
                    font.pixelSize: 10
                }
                Label {
                    text: editor.record.launch_target || ""
                    color: colors.muted
                    font.pixelSize: 10
                    Layout.fillWidth: true
                    wrapMode: Text.WrapAnywhere
                }
                Label {
                    visible: editor.record.available === false
                    text: "Цель недоступна. Проверьте путь и обновите каталог."
                    color: colors.danger
                    font.pixelSize: 12
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                }
                Label {
                    id: editError
                    color: colors.danger
                    font.pixelSize: 12
                    visible: text !== ""
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                }
            }
        }
        footer: Item {
            implicitHeight: 70
            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 24
                anchors.rightMargin: 24
                anchors.topMargin: 6
                anchors.bottomMargin: 24
                spacing: 8
                AppButton {
                    text: editor.record.source === "manual" ? "Удалить" : "Отключить"
                    dark: page.dark
                    destructive: true
                    onClicked: { controller.removeObject(editor.record.object_id); editor.close() }
                }
                Item { Layout.fillWidth: true }
                AppButton { text: "Отмена"; dark: page.dark; onClicked: editor.close() }
                AppButton {
                    objectName: "saveObjectButton"
                    text: "Сохранить"
                    primary: true
                    dark: page.dark
                    onClicked: {
                        if (controller.saveObject(editor.record.object_id, nameField.text,
                                descriptionField.text, aliasesField.text, enabledBox.checked, favoriteBox.checked))
                            editor.close()
                        else
                            editError.text = "Проверьте название и попробуйте ещё раз."
                    }
                }
            }
        }
    }
}
