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

    RowLayout {
        Layout.fillWidth: true
        ColumnLayout {
            spacing: 4
            Label { text: "Ваш каталог"; color: page.ink; font.pixelSize: 25; font.weight: Font.DemiBold }
            Label { text: controller.totalCount + " записей · " + controller.activeCount + " доступно"; color: page.muted; font.pixelSize: 12 }
        }
        Item { Layout.fillWidth: true }
        AppButton { text: "+ Файл"; dark: page.dark; onClicked: controller.addPath(false) }
        AppButton { text: "+ Папка"; dark: page.dark; onClicked: controller.addPath(true) }
        AppButton {
            objectName: "refreshButton"
            text: controller.scanning ? "Собираем…" : "Обновить"
            dark: page.dark; primary: true; enabled: !controller.scanning
            onClicked: controller.refresh()
        }
    }
    Field {
        objectName: "catalogSearchField"
        Layout.fillWidth: true
        dark: page.dark
        placeholderText: "Найти запись, включая отключённые"
        onTextEdited: controller.setCatalogQuery(text)
    }
    Rectangle {
        Layout.fillWidth: true
        Layout.fillHeight: true
        color: page.panel
        radius: 12
        border.color: page.line
        ListView {
            id: catalogList
            objectName: "catalogList"
            anchors.fill: parent
            anchors.margins: 8
            clip: true
            spacing: 3
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar {}
            model: controller.catalogItems
            delegate: Rectangle {
                id: row
                required property var modelData
                width: catalogList.width
                height: 65
                radius: 8
                color: rowMouse.containsMouse ? (page.dark ? "#222f42" : "#f0f4fa") : "transparent"
                MouseArea {
                    id: rowMouse
                    anchors.fill: parent
                    hoverEnabled: true
                    onDoubleClicked: page.editObject(row.modelData)
                }
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 10; anchors.rightMargin: 10
                    spacing: 12
                    Image {
                        source: row.modelData.icon
                        asynchronous: false
                        Layout.preferredWidth: 30; Layout.preferredHeight: 30
                        sourceSize: Qt.size(30, 30)
                        fillMode: Image.PreserveAspectFit
                        opacity: row.modelData.enabled && row.modelData.available ? 1 : 0.45
                    }
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 4
                        Label {
                            text: row.modelData.name
                            color: row.modelData.enabled && row.modelData.available ? page.ink : page.muted
                            font.pixelSize: 14
                            Layout.fillWidth: true; elide: Text.ElideRight
                        }
                        Label {
                            text: !row.modelData.available ? "Недоступен · " + row.modelData.sourceLabel
                                 : !row.modelData.enabled ? "Отключён · " + row.modelData.sourceLabel
                                 : row.modelData.description || row.modelData.sourceLabel
                            color: page.muted; font.pixelSize: 12
                            Layout.fillWidth: true; elide: Text.ElideRight
                        }
                    }
                    Label { text: row.modelData.favorite ? "★" : ""; color: "#d7bb74" }
                    AppButton {
                        text: "Править"; compact: true; dark: page.dark
                        onClicked: page.editObject(row.modelData)
                    }
                }
            }
        }
        Label {
            anchors.centerIn: parent
            visible: catalogList.count === 0
            text: "Здесь появятся ваши приложения и папки"
            color: page.muted; font.pixelSize: 15
        }
    }
    RowLayout {
        Layout.fillWidth: true
        Label { text: "Обновление: " + controller.lastScan; color: page.muted; font.pixelSize: 11 }
        Item { Layout.fillWidth: true }
        AppButton {
            text: "Экспорт каталога"; compact: true; dark: page.dark
            enabled: controller.activeCount > 0
            onClicked: controller.exportCatalog()
        }
    }
    Dialog {
        id: editor
        objectName: "objectEditor"
        parent: Overlay.overlay
        width: Math.min(590, parent.width - 48)
        height: Math.min(540, parent.height - 48)
        x: (parent.width - width) / 2
        y: (parent.height - height) / 2
        modal: true
        property var record: ({})
        padding: 20
        closePolicy: Popup.CloseOnEscape
        background: Rectangle { color: page.panel; radius: 14; border.color: page.line }
        header: Label {
            text: "Описание объекта"; color: page.ink
            font.pixelSize: 20; font.weight: Font.DemiBold
            padding: 20; bottomPadding: 0
        }
        contentItem: ScrollView {
            clip: true
            ColumnLayout {
                width: editor.availableWidth
                spacing: 9
                Label { text: "Название"; color: page.muted; font.pixelSize: 12 }
                Field { id: nameField; objectName: "nameField"; dark: page.dark; Layout.fillWidth: true }
                Label { text: "Для чего этот объект"; color: page.muted; font.pixelSize: 12 }
                TextArea {
                    id: descriptionField
                    objectName: "descriptionField"
                    Layout.fillWidth: true
                    Layout.preferredHeight: 84
                    wrapMode: TextEdit.Wrap
                    color: page.ink
                    placeholderText: "Например: папка проекта по машинному обучению"
                    placeholderTextColor: page.muted
                    font.pixelSize: 14
                    padding: 12
                    background: Rectangle {
                        radius: 8
                        color: page.dark ? "#151e2c" : "#ffffff"
                        border.color: descriptionField.activeFocus ? "#83adff" : page.line
                    }
                }
                Label { text: "Другие названия через запятую"; color: page.muted; font.pixelSize: 12 }
                Field { id: aliasesField; objectName: "aliasesField"; dark: page.dark; Layout.fillWidth: true }
                RowLayout {
                    CheckBox { id: enabledBox; text: "Показывать в поиске"; palette.windowText: page.ink }
                    CheckBox { id: favoriteBox; text: "Избранное"; palette.windowText: page.ink }
                }
                Label {
                    text: (editor.record.kindLabel || "") + " · " + (editor.record.sourceLabel || "")
                    color: page.muted; font.pixelSize: 11
                }
                Label {
                    text: editor.record.launch_target || ""
                    color: page.muted; font.pixelSize: 11
                    Layout.fillWidth: true; wrapMode: Text.WrapAnywhere
                }
                Label {
                    visible: editor.record.available === false
                    text: "Цель недоступна. Проверьте путь и обновите каталог."
                    color: "#efaaaa"; font.pixelSize: 12
                }
                Label { id: editError; color: "#efaaaa"; font.pixelSize: 12; visible: text !== "" }
            }
        }
        footer: RowLayout {
            spacing: 8
            Item { Layout.preferredWidth: 12 }
            AppButton {
                text: editor.record.source === "manual" ? "Удалить" : "Отключить"
                dark: page.dark
                onClicked: { controller.removeObject(editor.record.object_id); editor.close() }
            }
            Item { Layout.fillWidth: true }
            AppButton { text: "Отмена"; dark: page.dark; onClicked: editor.close() }
            AppButton {
                objectName: "saveObjectButton"
                text: "Сохранить"; primary: true
                onClicked: {
                    if (controller.saveObject(editor.record.object_id, nameField.text,
                            descriptionField.text, aliasesField.text, enabledBox.checked, favoriteBox.checked))
                        editor.close()
                    else
                        editError.text = "Проверьте название и попробуйте ещё раз."
                }
            }
            Item { Layout.preferredWidth: 12 }
        }
    }
}
