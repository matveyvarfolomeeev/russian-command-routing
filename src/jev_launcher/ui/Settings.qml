import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ScrollView {
    id: page
    objectName: "settingsPage"
    property bool dark: true
    property color ink
    property color muted
    property color panel
    property color line
    clip: true
    contentWidth: availableWidth

    ColumnLayout {
        width: page.availableWidth
        spacing: 16
        Label { text: "Настройки"; color: page.ink; font.pixelSize: 25; font.weight: Font.DemiBold }
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: appearance.implicitHeight + 36
            color: page.panel; radius: 12; border.color: page.line
            ColumnLayout {
                id: appearance
                anchors.fill: parent; anchors.margins: 18
                spacing: 10
                Label { text: "Быстрый доступ"; color: page.ink; font.pixelSize: 16 }
                Label { text: "Горячая клавиша"; color: page.muted; font.pixelSize: 12 }
                Field {
                    id: hotkeyField; objectName: "hotkeyField"; dark: page.dark
                    Layout.fillWidth: true; text: controller.hotkeyText
                    placeholderText: "Ctrl+Alt+Space"
                }
                RowLayout {
                    Label { text: "Тема"; color: page.muted; font.pixelSize: 12 }
                    Item { Layout.fillWidth: true }
                    ComboBox {
                        id: themeBox
                        objectName: "themeBox"
                        model: ["Тёмная", "Светлая"]
                        currentIndex: controller.theme === "dark" ? 0 : 1
                        implicitWidth: 150
                    }
                }
            }
        }
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: voiceSettings.implicitHeight + 36
            color: page.panel; radius: 12; border.color: page.line
            ColumnLayout {
                id: voiceSettings
                anchors.fill: parent; anchors.margins: 18
                spacing: 10
                Label { text: "Голосовой ввод"; color: page.ink; font.pixelSize: 16 }
                Label {
                    text: "Whisper Large V3 Turbo через Groq. Запись отправляется после нажатия «Стоп»."
                    color: page.muted; font.pixelSize: 12
                    Layout.fillWidth: true; wrapMode: Text.Wrap
                }
                Label { text: "Микрофон"; color: page.muted; font.pixelSize: 12 }
                ComboBox {
                    id: microphoneBox; objectName: "microphoneBox"
                    Layout.fillWidth: true
                    model: controller.microphones
                    textRole: "name"
                    valueRole: "id"
                    Component.onCompleted: currentIndex = Math.max(0, indexOfValue(controller.microphoneId))
                }
                Label {
                    text: controller.keyConfigured ? "API-ключ настроен" : "Добавьте API-ключ Groq"
                    color: controller.keyConfigured ? "#8cc9ae" : page.muted
                    font.pixelSize: 12
                }
                Field {
                    id: keyField; objectName: "apiKeyField"; dark: page.dark
                    Layout.fillWidth: true
                    echoMode: TextInput.Password
                    placeholderText: "Новый ключ — оставить пустым, чтобы сохранить прежний"
                }
                CheckBox { id: clearKey; text: "Удалить сохранённый ключ"; palette.windowText: page.ink }
                Label {
                    text: "Ключ хранится локально с защитой Windows. GROQ_API_KEY из окружения имеет приоритет."
                    color: page.muted; font.pixelSize: 11
                    Layout.fillWidth: true; wrapMode: Text.Wrap
                }
            }
        }
        RowLayout {
            Layout.fillWidth: true
            Label {
                text: "Записи голоса не сохраняются."
                color: page.muted; font.pixelSize: 12
                Layout.fillWidth: true
            }
            AppButton {
                objectName: "saveSettingsButton"
                text: "Сохранить настройки"; primary: true
                onClicked: {
                    if (controller.saveSettings(hotkeyField.text, themeBox.currentIndex === 0 ? "dark" : "light",
                            microphoneBox.currentValue || "", keyField.text, clearKey.checked)) {
                        keyField.clear()
                        clearKey.checked = false
                    }
                }
            }
        }
        Item { height: 6 }
    }
}
