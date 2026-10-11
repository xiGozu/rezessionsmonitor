/*++

DiyHid – virtuelle Maus + Tastatur für Windows 10/11 (KMDF + Virtual HID Framework)

Der Treiber meldet über VHF ein HID-Gerät mit zwei Top-Level-Collections an:
  Report-ID 1: Maus     (3 Tasten, X, Y, Rad – relativ)
  Report-ID 2: Tastatur (Modifier, reserviert, 6 Tastencodes – Boot-Layout)
Windows bindet dafür die normalen Inbox-Treiber (mouhid.sys / kbdhid.sys) –
für das System ist es also eine ganz normale HID-Maus und -Tastatur.

Anwendungen schicken Reports per DeviceIoControl an \\.\DiyHid.
Wird das Handle geschlossen (auch wenn das Programm abstürzt), lässt der
Treiber alle Tasten los.

--*/

#include <ntddk.h>
#include <wdf.h>
#include <vhf.h>

#define DIYHID_IOCTL_SUBMIT_REPORT \
    CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800, METHOD_BUFFERED, FILE_WRITE_DATA)

#define REPORT_ID_MOUSE        1
#define REPORT_ID_KEYBOARD     2
#define MOUSE_REPORT_LEN       5    // ID, Tasten, X, Y, Rad
#define KEYBOARD_REPORT_LEN    9    // ID, Modifier, reserviert, 6 Tastencodes
#define KEYBOARD_MAX_USAGE     0x65

static UCHAR g_ReportDescriptor[] = {
    // ---- Maus ----
    0x05, 0x01,        // Usage Page (Generic Desktop)
    0x09, 0x02,        // Usage (Mouse)
    0xA1, 0x01,        // Collection (Application)
    0x85, REPORT_ID_MOUSE,
    0x09, 0x01,        //   Usage (Pointer)
    0xA1, 0x00,        //   Collection (Physical)
    0x05, 0x09,        //     Usage Page (Button)
    0x19, 0x01,        //     Usage Minimum (1)
    0x29, 0x03,        //     Usage Maximum (3)
    0x15, 0x00,        //     Logical Minimum (0)
    0x25, 0x01,        //     Logical Maximum (1)
    0x95, 0x03,        //     Report Count (3)
    0x75, 0x01,        //     Report Size (1)
    0x81, 0x02,        //     Input (Data, Var, Abs)
    0x95, 0x01,        //     Report Count (1)
    0x75, 0x05,        //     Report Size (5)
    0x81, 0x03,        //     Input (Const) – Auffüllbits
    0x05, 0x01,        //     Usage Page (Generic Desktop)
    0x09, 0x30,        //     Usage (X)
    0x09, 0x31,        //     Usage (Y)
    0x09, 0x38,        //     Usage (Wheel)
    0x15, 0x81,        //     Logical Minimum (-127)
    0x25, 0x7F,        //     Logical Maximum (127)
    0x75, 0x08,        //     Report Size (8)
    0x95, 0x03,        //     Report Count (3)
    0x81, 0x06,        //     Input (Data, Var, Rel)
    0xC0,              //   End Collection
    0xC0,              // End Collection

    // ---- Tastatur ----
    0x05, 0x01,        // Usage Page (Generic Desktop)
    0x09, 0x06,        // Usage (Keyboard)
    0xA1, 0x01,        // Collection (Application)
    0x85, REPORT_ID_KEYBOARD,
    0x05, 0x07,        //   Usage Page (Keyboard/Keypad)
    0x19, 0xE0,        //   Usage Minimum (Left Control)
    0x29, 0xE7,        //   Usage Maximum (Right GUI)
    0x15, 0x00,        //   Logical Minimum (0)
    0x25, 0x01,        //   Logical Maximum (1)
    0x75, 0x01,        //   Report Size (1)
    0x95, 0x08,        //   Report Count (8)
    0x81, 0x02,        //   Input (Data, Var, Abs) – Modifier
    0x75, 0x08,        //   Report Size (8)
    0x95, 0x01,        //   Report Count (1)
    0x81, 0x03,        //   Input (Const) – reserviert
    0x05, 0x07,        //   Usage Page (Keyboard/Keypad)
    0x19, 0x00,        //   Usage Minimum (0)
    0x29, KEYBOARD_MAX_USAGE,
    0x15, 0x00,        //   Logical Minimum (0)
    0x25, KEYBOARD_MAX_USAGE,
    0x75, 0x08,        //   Report Size (8)
    0x95, 0x06,        //   Report Count (6)
    0x81, 0x00,        //   Input (Data, Array)
    0xC0,              // End Collection
};

typedef struct _DEVICE_CONTEXT {
    VHFHANDLE VhfHandle;
} DEVICE_CONTEXT, *PDEVICE_CONTEXT;

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(DEVICE_CONTEXT, DeviceGetContext)

DRIVER_INITIALIZE DriverEntry;
EVT_WDF_DRIVER_DEVICE_ADD DiyHidEvtDeviceAdd;
EVT_WDF_OBJECT_CONTEXT_CLEANUP DiyHidEvtDeviceCleanup;
EVT_WDF_FILE_CLEANUP DiyHidEvtFileCleanup;
EVT_WDF_IO_QUEUE_IO_DEVICE_CONTROL DiyHidEvtIoDeviceControl;

#ifdef ALLOC_PRAGMA
#pragma alloc_text(INIT, DriverEntry)
#pragma alloc_text(PAGE, DiyHidEvtDeviceAdd)
#pragma alloc_text(PAGE, DiyHidEvtDeviceCleanup)
#pragma alloc_text(PAGE, DiyHidEvtFileCleanup)
#endif

static NTSTATUS
DiyHidSubmit(
    _In_ PDEVICE_CONTEXT Context,
    _In_reads_bytes_(Length) PUCHAR Report,
    _In_ ULONG Length
    )
{
    HID_XFER_PACKET packet;

    if (Context->VhfHandle == NULL) {
        return STATUS_DEVICE_NOT_READY;
    }

    // Report-IDs werden benutzt, daher steht die ID im ersten Byte des Puffers.
    packet.reportBuffer = Report;
    packet.reportBufferLen = Length;
    packet.reportId = Report[0];

    // VHF puffert den Report selbst, der Puffer darf danach wiederverwendet werden.
    return VhfReadReportSubmit(Context->VhfHandle, &packet);
}

static VOID
DiyHidReleaseAll(
    _In_ PDEVICE_CONTEXT Context
    )
{
    UCHAR mouse[MOUSE_REPORT_LEN] = { REPORT_ID_MOUSE };
    UCHAR keyboard[KEYBOARD_REPORT_LEN] = { REPORT_ID_KEYBOARD };

    (VOID)DiyHidSubmit(Context, mouse, sizeof(mouse));
    (VOID)DiyHidSubmit(Context, keyboard, sizeof(keyboard));
}

NTSTATUS
DriverEntry(
    _In_ PDRIVER_OBJECT DriverObject,
    _In_ PUNICODE_STRING RegistryPath
    )
{
    WDF_DRIVER_CONFIG config;

    WDF_DRIVER_CONFIG_INIT(&config, DiyHidEvtDeviceAdd);
    return WdfDriverCreate(DriverObject, RegistryPath, WDF_NO_OBJECT_ATTRIBUTES,
                           &config, WDF_NO_HANDLE);
}

NTSTATUS
DiyHidEvtDeviceAdd(
    _In_ WDFDRIVER Driver,
    _Inout_ PWDFDEVICE_INIT DeviceInit
    )
{
    NTSTATUS status;
    WDFDEVICE device;
    PDEVICE_CONTEXT context;
    WDF_OBJECT_ATTRIBUTES attributes;
    WDF_FILEOBJECT_CONFIG fileConfig;
    WDF_IO_QUEUE_CONFIG queueConfig;
    VHF_CONFIG vhfConfig;
    DECLARE_CONST_UNICODE_STRING(symbolicLink, L"\\DosDevices\\DiyHid");

    UNREFERENCED_PARAMETER(Driver);
    PAGED_CODE();

    WDF_FILEOBJECT_CONFIG_INIT(&fileConfig, WDF_NO_EVENT_CALLBACK,
                               WDF_NO_EVENT_CALLBACK, DiyHidEvtFileCleanup);
    WdfDeviceInitSetFileObjectConfig(DeviceInit, &fileConfig, WDF_NO_OBJECT_ATTRIBUTES);

    WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&attributes, DEVICE_CONTEXT);
    attributes.EvtCleanupCallback = DiyHidEvtDeviceCleanup;

    status = WdfDeviceCreate(&DeviceInit, &attributes, &device);
    if (!NT_SUCCESS(status)) {
        return status;
    }

    context = DeviceGetContext(device);
    context->VhfHandle = NULL;

    status = WdfDeviceCreateSymbolicLink(device, &symbolicLink);
    if (!NT_SUCCESS(status)) {
        return status;
    }

    // Sequenziell: Reports kommen in genau der Reihenfolge an, in der sie gesendet wurden.
    WDF_IO_QUEUE_CONFIG_INIT_DEFAULT_QUEUE(&queueConfig, WdfIoQueueDispatchSequential);
    queueConfig.EvtIoDeviceControl = DiyHidEvtIoDeviceControl;
    status = WdfIoQueueCreate(device, &queueConfig, WDF_NO_OBJECT_ATTRIBUTES, WDF_NO_HANDLE);
    if (!NT_SUCCESS(status)) {
        return status;
    }

    VHF_CONFIG_INIT(&vhfConfig, WdfDeviceWdmGetDeviceObject(device),
                    (USHORT)sizeof(g_ReportDescriptor), g_ReportDescriptor);

    status = VhfCreate(&vhfConfig, &context->VhfHandle);
    if (!NT_SUCCESS(status)) {
        context->VhfHandle = NULL;
        return status;
    }

    return VhfStart(context->VhfHandle);
}

VOID
DiyHidEvtDeviceCleanup(
    _In_ WDFOBJECT Object
    )
{
    PDEVICE_CONTEXT context;

    PAGED_CODE();

    context = DeviceGetContext((WDFDEVICE)Object);
    if (context->VhfHandle != NULL) {
        VhfDelete(context->VhfHandle, TRUE);
        context->VhfHandle = NULL;
    }
}

VOID
DiyHidEvtFileCleanup(
    _In_ WDFFILEOBJECT FileObject
    )
{
    PAGED_CODE();

    DiyHidReleaseAll(DeviceGetContext(WdfFileObjectGetDevice(FileObject)));
}

static BOOLEAN
DiyHidIsValidReport(
    _In_reads_bytes_(Length) PUCHAR Report,
    _In_ size_t Length
    )
{
    size_t i;

    if (Report[0] == REPORT_ID_MOUSE) {
        return Length == MOUSE_REPORT_LEN && (Report[1] & ~0x07) == 0;
    }

    if (Report[0] == REPORT_ID_KEYBOARD && Length == KEYBOARD_REPORT_LEN) {
        for (i = 3; i < KEYBOARD_REPORT_LEN; i++) {
            if (Report[i] > KEYBOARD_MAX_USAGE) {
                return FALSE;
            }
        }
        return TRUE;
    }

    return FALSE;
}

VOID
DiyHidEvtIoDeviceControl(
    _In_ WDFQUEUE Queue,
    _In_ WDFREQUEST Request,
    _In_ size_t OutputBufferLength,
    _In_ size_t InputBufferLength,
    _In_ ULONG IoControlCode
    )
{
    NTSTATUS status;
    PVOID buffer;
    size_t length;
    UCHAR report[KEYBOARD_REPORT_LEN];

    UNREFERENCED_PARAMETER(OutputBufferLength);
    UNREFERENCED_PARAMETER(InputBufferLength);

    if (IoControlCode != DIYHID_IOCTL_SUBMIT_REPORT) {
        WdfRequestComplete(Request, STATUS_INVALID_DEVICE_REQUEST);
        return;
    }

    status = WdfRequestRetrieveInputBuffer(Request, 1, &buffer, &length);
    if (NT_SUCCESS(status)) {
        if (length > sizeof(report)) {
            status = STATUS_INVALID_BUFFER_SIZE;
        } else {
            // Eigene Kopie: der Systempuffer gehört dem Request.
            RtlCopyMemory(report, buffer, length);
            if (!DiyHidIsValidReport(report, length)) {
                status = STATUS_INVALID_PARAMETER;
            } else {
                status = DiyHidSubmit(DeviceGetContext(WdfIoQueueGetDevice(Queue)),
                                      report, (ULONG)length);
            }
        }
    }

    WdfRequestComplete(Request, status);
}
