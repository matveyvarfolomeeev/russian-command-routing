import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ColumnLayout {
    id: page
    objectName: "settingsPage"
    property bool dark: true
    property color ink
    property color muted
    property color panel
    property color line
    Theme { id: colors; dark: page.dark }
    spacing: 14

    Label {
        text: "Настройки"
        color: colors.ink
        font.pixelSize: 25
        font.weight: Font.DemiBold
        font.letterSpacing: -0.6
    }
    ScrollView {
        id: settingsScroll
        Layout.fillWidth: true
        Layout.fillHeight: true
        clip: true
        contentWidth: availableWidth
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
        ColumnLayout {
            width: settingsScroll.availableWidth
            spacing: 12
            Rectangle {
                Layout.fillWidth: true
                implicitHeight: appearance.implicitHeight + 24
                color: colors.surface
                radius: 11
                border.color: colors.line
                ColumnLayout {
                    id: appearance
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 6
                    Label { text: "Быстрый доступ"; color: colors.ink; font.pixelSize: 15; font.weight: Font.DemiBold }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 16
                        Label { text: "Горячая клавиша"; color: colors.muted; font.pixelSize: 12; Layout.preferredWidth: 145 }
                        Field {
                            id: hotkeyField
                            objectName: "hotkeyField"
                            dark: page.dark
                            Layout.fillWidth: true
                            text: controller.hotkeyText
                            placeholderText: "Ctrl+Shift+Space"
                            implicitHeight: 38
                        }
                    }
                    Label {
                        text: controller.hotkeyStatus
                        color: controller.hotkeyRegistered ? colors.muted : colors.danger
                        font.pixelSize: 11
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 16
                        Label { text: "Оформление"; color: colors.muted; font.pixelSize: 12; Layout.preferredWidth: 145 }
                        AppCombo {
                            id: themeBox
                            objectName: "themeBox"
                            dark: page.dark
                            model: ["Тёмная", "Светлая"]
                            currentIndex: controller.theme === "dark" ? 0 : 1
                            Layout.fillWidth: true
                            implicitHeight: 38
                        }
                    }
                }
            }
            Rectangle {
                Layout.fillWidth: true
                implicitHeight: voiceSettings.implicitHeight + 24
                color: colors.surface
                radius: 11
                border.color: colors.line
                ColumnLayout {
                    id: voiceSettings
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 6
                    Label { text: "Голосовой ввод"; color: colors.ink; font.pixelSize: 15; font.weight: Font.DemiBold }
                    Label {
                        text: "Запись распознаётся через Groq после нажатия «Стоп»."
                        color: colors.muted
                        font.pixelSize: 12
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 16
                        Label { text: "Микрофон"; color: colors.muted; font.pixelSize: 12; Layout.preferredWidth: 145 }
                        AppCombo {
                            id: microphoneBox
                            objectName: "microphoneBox"
                            dark: page.dark
                            Layout.fillWidth: true
                            model: controller.microphones
                            textRole: "name"
                            valueRole: "id"
                            implicitHeight: 38
                            Component.onCompleted: currentIndex = Math.max(0, indexOfValue(controller.microphoneId))
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 16
                        Label { text: "API-ключ Groq"; color: colors.muted; font.pixelSize: 12; Layout.preferredWidth: 145 }
                        Field {
                            id: keyField
                            objectName: "apiKeyField"
                            dark: page.dark
                            Layout.fillWidth: true
                            echoMode: TextInput.Password
                            placeholderText: "Оставьте пустым, чтобы сохранить прежний"
                            implicitHeight: 38
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12
                        Label {
                            text: controller.keyConfigured ? "Ключ настроен" : "Ключ ещё не добавлен"
                            color: controller.keyConfigured ? colors.accent : colors.subtle
                            font.pixelSize: 11
                        }
                        Item { Layout.fillWidth: true }
                        AppCheck { id: clearKey; text: "Удалить сохранённый ключ"; dark: page.dark; implicitHeight: 28 }
                    }
                    Label {
                        text: "Ключ защищён Windows. Аудиозаписи не сохраняются."
                        color: colors.subtle
                        font.pixelSize: 11
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                    }
                }
            }
            Item { implicitHeight: 3 }
        }
    }
    RowLayout {
        Layout.fillWidth: true
        spacing: 12
        Label {
            text: "Изменения применятся после сохранения"
            color: colors.subtle
            font.pixelSize: 11
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            elide: Text.ElideRight
        }
        AppButton {
            objectName: "saveSettingsButton"
            text: "Сохранить настройки"
            primary: true
            dark: page.dark
            onClicked: {
                if (controller.saveSettings(hotkeyField.text, themeBox.currentIndex === 0 ? "dark" : "light",
                        microphoneBox.currentValue || "", keyField.text, clearKey.checked)) {
                    keyField.clear()
                    clearKey.checked = false
                }
            }
        }
    }
}
