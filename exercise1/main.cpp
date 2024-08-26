#include <volk/volk.h>

#include <iostream>

#include <string>
#include <vector>
#include <optional>
#include <unordered_set>

#include <cstdio>
#include <cassert>
#include <cstdint>
#include <cstring>

#include "../labutils/to_string.hpp"
namespace lut = labutils;

namespace
{
std::unordered_set<std::string> get_instance_layers();
std::unordered_set<std::string> get_instance_extensions();

VkInstance create_instance(std::vector<char const*> const& enabledLayers = {},
						   std::vector<char const*> const& enabledInstanceExtensions = {}, bool aEnableDebugUtils = false);

void enumerate_devices(VkInstance);

float score_device(VkPhysicalDevice);
VkPhysicalDevice select_device(VkInstance);

std::optional<std::uint32_t> find_graphics_queue_family(VkPhysicalDevice);
VkDevice create_device(VkPhysicalDevice, std::uint32_t queueFamily);

VkDebugUtilsMessengerEXT create_debug_messenger(VkInstance);
VKAPI_ATTR VkBool32 VKAPI_CALL debug_util_callback(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
												   VkDebugUtilsMessengerCallbackDataEXT const* data, void* userPtr);
}

int main()
{
	// Use Volk to load the initial parts of the Vulkan API that are required
	// to create a Vulkan instance. This includes very few functions:
	// - vkGetInstanceProcAddr()
	// - vkCreateInstance()
	// - vkEnumerateInstanceExtensionProperties()
	// - vkEnumerateInstanceLayerProperties()
	// - vkEnumerateInstanceVersion() (added in Vulkan 1.1)
	
	if (auto const res = volkInitialize(); res != VK_SUCCESS)
	{
		std::fprintf(stderr, "Error: unable to load Vulkan API\n");
		std::fprintf(stderr, "Volk returned error %s\n", lut::to_string(res).c_str());
		return 1;
	}

	// We can use vkEnumerateInstanceVersion() to tell us the version of the
	// Vulkan loader. vkEnumerateInstanceVersion() was added in Vulkan 1.1, so
	// it might not be available on systems with particularly old Vulkan
	// loaders. In that case, assume the version is 1.0.x and output this.
	std::uint32_t loaderVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
	if (vkEnumerateInstanceVersion)
	{
		if (auto const res = vkEnumerateInstanceVersion(&loaderVersion); res != VK_SUCCESS)
		{
			std::fprintf(stderr, "Warning: vkEnumerateInstanceVersion() returned error %s\n", lut::to_string(res).c_str());
		}
	}

	std::printf("Vulkan loader version: %d.%d.%d (variant %d)\n", VK_API_VERSION_MAJOR(loaderVersion), VK_API_VERSION_MINOR(loaderVersion), VK_API_VERSION_PATCH(loaderVersion), VK_API_VERSION_VARIANT(loaderVersion));

	// Check instance layers and extensions
	auto const supportedLayers = get_instance_layers();
	auto const supportedExtensions = get_instance_extensions();

	std::vector<char const*> enabledLayers;
	std::vector<char const*> enabledExtensions;
	bool enableDebugUtils = false;

#if !defined(NDEBUG) // debug builds only
	if(supportedLayers.count("VK_LAYER_KHRONOS_validation"))
	{
		enabledLayers.emplace_back("VK_LAYER_KHRONOS_validation");
	}

	if(supportedExtensions.count("VK_EXT_debug_utils"))
	{
		enableDebugUtils = true;
		enabledExtensions.emplace_back("VK_EXT_debug_utils");
	}
#endif // ∼ debug builds

	for (auto const& layer : enabledLayers)
	{
		std::printf("Enabling layer: %s\n", layer);
	}
	for (auto const& extension : enabledExtensions)
	{
		std::printf("Enabling extension: %s\n", extension);
	}

	// Create Vulkan instance
	VkInstance instance = create_instance(enabledLayers, enabledExtensions, enableDebugUtils);
	if(VK_NULL_HANDLE == instance)
		return 1;

	// Instruct Volk to load the remainder of the Vulkan API.
	volkLoadInstance(instance);

	// Setup debug messenger
	VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
	if(enableDebugUtils)
	{
		debugMessenger = create_debug_messenger(instance);
	}

	// Print Vulkan devices
	enumerate_devices(instance);

	// Select appropriate Vulkan device
	VkPhysicalDevice physicalDevice = select_device(instance);
	if(VK_NULL_HANDLE == physicalDevice)
	{
		vkDestroyInstance(instance, nullptr);

		std::fprintf(stderr, "Error: no suitable physical device found!\n");
		return 1;
	}

	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(physicalDevice, &props);
	std::printf("Selected device: %s\n", props.deviceName);

	// Create a Logical Device
	auto const graphicsFamilyIndex = find_graphics_queue_family(physicalDevice);
	if (!graphicsFamilyIndex)
	{
		vkDestroyInstance(instance, nullptr);

		std::fprintf(stderr, "Error: no graphics queue found!\n");
		return 1;
	}

	VkDevice device = create_device(physicalDevice, *graphicsFamilyIndex);
	if (device == VK_NULL_HANDLE)
	{
		vkDestroyInstance(instance, nullptr);
		return 1;
	}
	volkLoadDevice(device);

	// Retrieve Graphics Queue
	VkQueue graphicsQueue = VK_NULL_HANDLE;
	vkGetDeviceQueue(device, *graphicsFamilyIndex, 0, &graphicsQueue);

	assert(graphicsQueue != VK_NULL_HANDLE);

	// Cleanup
	vkDestroyDevice(device, nullptr);
	if (debugMessenger != VK_NULL_HANDLE) {
		vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);}
	vkDestroyInstance(instance, nullptr);
	
	return 0;
}

namespace
{
std::unordered_set<std::string> get_instance_layers()
{
	std::uint32_t numLayers = 0;
	if (auto const res = vkEnumerateInstanceLayerProperties(&numLayers, nullptr); res != VK_SUCCESS)
	{
		std::fprintf(stderr, "Error: unable to enumerate layers\n");
		std::fprintf(stderr, "vkEnumerateInstanceLayerProperties() returned %s\n", lut::to_string(res).c_str());

		return {};
	}

	std::vector<VkLayerProperties> layers(numLayers);
	if (auto const res = vkEnumerateInstanceLayerProperties(&numLayers, layers.data()); res != VK_SUCCESS)
	{
		std::fprintf(stderr, "Error: unable to get layer properties\n");
		std::fprintf(stderr, "vkEnumerateInstanceLayerProperties() returned %s\n", lut::to_string(res).c_str());

		return {};
	}

	std::unordered_set<std::string> layerNames;
	for (auto const& layer : layers)
	{
		layerNames.insert(layer.layerName);
	}
	return layerNames;
}
std::unordered_set<std::string> get_instance_extensions()
{
	std::uint32_t numExtensions = 0;
	if (auto const res = vkEnumerateInstanceExtensionProperties(nullptr, &numExtensions, nullptr); res != VK_SUCCESS)
	{
		std::fprintf(stderr, "Error: unable to enumerate extensions\n");
		std::fprintf(stderr, "vkEnumerateInstanceExtensionProperties() returned %s\n", lut::to_string(res).c_str());

		return {};
	}

	std::vector<VkExtensionProperties> extensions(numExtensions);
	if (auto const res = vkEnumerateInstanceExtensionProperties(nullptr, &numExtensions, extensions.data()); res != VK_SUCCESS)
	{
		std::fprintf(stderr, "Error: unable to get extension properties\n");
		std::fprintf(stderr, "vkEnumerateInstanceExtensionProperties() returned %s\n", lut::to_string(res).c_str());

		return {};
	}

	std::unordered_set<std::string> extensionNames;
	for (auto const& extension : extensions)
	{
		extensionNames.insert(extension.extensionName);
	}
	return extensionNames;
}

VkInstance create_instance(std::vector<char const*> const& enabledLayers,
						   std::vector<char const*> const& enabledInstanceExtensions, bool enableDebugUtils)
{
	VkApplicationInfo appInfo{}; {
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "COMP5822-EX1.1";
		appInfo.applicationVersion = 2022;
		appInfo.apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);
	}
	VkInstanceCreateInfo instanceInfo{}; {
		instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instanceInfo.pApplicationInfo = &appInfo;

		instanceInfo.enabledLayerCount = std::uint32_t(enabledLayers.size());
		instanceInfo.ppEnabledLayerNames = enabledLayers.data();

		instanceInfo.enabledExtensionCount = std::uint32_t(enabledInstanceExtensions.size());
		instanceInfo.ppEnabledExtensionNames = enabledInstanceExtensions.data();
	}
	VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
	if (enableDebugUtils) {
		debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		debugInfo.pNext = instanceInfo.pNext;
		instanceInfo.pNext = &debugInfo;

		debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
									VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
								VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
								VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

		debugInfo.pfnUserCallback = &debug_util_callback;
		debugInfo.pUserData = nullptr;
	}

	VkInstance instance = VK_NULL_HANDLE;
	if (auto const res = vkCreateInstance(&instanceInfo, nullptr, &instance); res != VK_SUCCESS)
	{
		std::fprintf(stderr, "Error: unable to create Vulkan instance\n");
		std::fprintf(stderr, "vkCreateInstance(): %s\n", lut::to_string(res).c_str());

		return VK_NULL_HANDLE;
	}

	return instance;
}
}

namespace
{
float score_device(VkPhysicalDevice device)
{
	VkPhysicalDeviceProperties deviceProps;
	vkGetPhysicalDeviceProperties(device, &deviceProps);

	auto const versionMajor = VK_VERSION_MAJOR(deviceProps.apiVersion);
	auto const versionMinor = VK_VERSION_MINOR(deviceProps.apiVersion);
	if (versionMajor < 1 || (versionMajor == 1 && versionMinor < 2))
	{
		return -1.0f;
	}

	float deviceScore = 0.0f;
	if (deviceProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
		deviceScore += 500.0f;
	else if (deviceProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
		deviceScore += 100.0f;

	return deviceScore;
}
VkPhysicalDevice select_device(VkInstance instance)
{
	assert(instance != VK_NULL_HANDLE);

	float bestScore = -1.f;
	VkPhysicalDevice bestDevice = VK_NULL_HANDLE;

	std::uint32_t numDevices = 0;
	if (auto const res = vkEnumeratePhysicalDevices(instance, &numDevices, nullptr); res != VK_SUCCESS)
	{
		std::fprintf(stderr, "Error: unable to get phyiscal device count\n");
		std::fprintf(stderr, "vkEnumeratePhysicalDevices() returned error %s\n", lut::to_string(res).c_str());

		return VK_NULL_HANDLE;
	}
	std::vector<VkPhysicalDevice> devices(numDevices, VK_NULL_HANDLE);
	if (auto const res = vkEnumeratePhysicalDevices(instance, &numDevices, devices.data()); res != VK_SUCCESS)
	{
		std::fprintf(stderr, "Error: unable to get phyiscal device list\n");
		std::fprintf(stderr, "vkEnumeratePhysicalDevices() returned error %s\n", lut::to_string(res).c_str());

		return VK_NULL_HANDLE;
	}

	for(auto const device : devices)
	{
		auto const score = score_device(device);
		if(score > bestScore)
		{
			bestScore = score;
			bestDevice = device;
		}
	}

	return bestDevice;
}

void enumerate_devices(VkInstance instance)
{
	assert(instance != VK_NULL_HANDLE);

	std::uint32_t numDevices = 0;
	if (auto const res = vkEnumeratePhysicalDevices(instance, &numDevices, nullptr); res != VK_SUCCESS)
	{
		std::fprintf(stderr, "Error: unable to get phyiscal device count\n");
		std::fprintf(stderr, "vkEnumeratePhysicalDevices() returned error %s\n", lut::to_string(res).c_str());

		return;
	}
	std::vector<VkPhysicalDevice> devices(numDevices, VK_NULL_HANDLE);
	if (auto const res = vkEnumeratePhysicalDevices(instance, &numDevices, devices.data()); res != VK_SUCCESS)
	{
		std::fprintf(stderr, "Error: unable to get phyiscal device list\n");
		std::fprintf(stderr, "vkEnumeratePhysicalDevices() returned error %s\n", lut::to_string(res).c_str());

		return;
	}

	std::printf("Found %zu devices:\n", devices.size());
	for (auto const device : devices)
	{
		VkPhysicalDeviceProperties deviceProps;
		vkGetPhysicalDeviceProperties(device, &deviceProps);

		auto const versionMajor = VK_VERSION_MAJOR(deviceProps.apiVersion);
		auto const versionMinor = VK_VERSION_MINOR(deviceProps.apiVersion);
		auto const versionPatch = VK_VERSION_PATCH(deviceProps.apiVersion);

		std::printf("- %s (Vulkan: %d.%d.%d, Driver: %s)\n", deviceProps.deviceName, versionMajor, versionMinor, versionPatch,
					lut::driver_version(deviceProps.vendorID, deviceProps.driverVersion).c_str());
		std::printf(" - Type: %s\n", lut::to_string(deviceProps.deviceType).c_str());

		if (versionMajor > 1 || (versionMajor == 1 && versionMinor >= 1))
		{
			VkPhysicalDeviceFeatures2 deviceFeatures{}; {
				deviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
			}
			vkGetPhysicalDeviceFeatures2(device, &deviceFeatures);

			std::printf(" - Anisotropic filtering: %s\n", deviceFeatures.features.samplerAnisotropy ? "true" : "false");
		}

		std::uint32_t numQueues = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &numQueues, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(numQueues);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &numQueues, queueFamilies.data());

		for (auto const& family : queueFamilies)
		{
			std::printf(" - Queue family: %s (%u queues)\n", lut::queue_flags(family.queueFlags).c_str(), family.queueCount);
		}

		VkPhysicalDeviceMemoryProperties deviceMemProps;
		vkGetPhysicalDeviceMemoryProperties(device, &deviceMemProps);

		std::printf(" - %u heaps\n", deviceMemProps.memoryHeapCount);
		for (std::uint32_t i = 0; i < deviceMemProps.memoryHeapCount; ++i)
		{
			std::printf("  - heap %2u: %6zu MBytes, %s\n", i, std::size_t(deviceMemProps.memoryHeaps[i].size) / 1024 / 1024,
						lut::memory_heap_flags(deviceMemProps.memoryHeaps[i].flags).c_str());
		}

		std::printf(" - %u memory types\n", deviceMemProps.memoryTypeCount );
		for( std::uint32_t i = 0; i < deviceMemProps.memoryTypeCount; ++i )
		{
			std::printf("  - type %2u: from heap %2u, %s\n", i, deviceMemProps.memoryTypes[i].heapIndex,
						lut::memory_property_flags(deviceMemProps.memoryTypes[i].propertyFlags).c_str());
		}
	}
}

std::optional<std::uint32_t> find_graphics_queue_family(VkPhysicalDevice device)
{
	assert(device != VK_NULL_HANDLE);

	std::uint32_t numQueues = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &numQueues, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilies(numQueues);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &numQueues, queueFamilies.data());

	for (std::uint32_t i = 0; i < numQueues; i++)
	{
		auto const& family = queueFamilies[i];

		if (family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			return i;
	}

	return {};
}
VkDevice create_device(VkPhysicalDevice physicalDevice, std::uint32_t queueFamily)
{
	assert(physicalDevice != VK_NULL_HANDLE);

	//float queueProperties[1] = {1.0f};
	float queueProperties = 1.0f;

	VkDeviceQueueCreateInfo queueInfo{}; {
		queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;

		queueInfo.queueFamilyIndex = queueFamily;
		queueInfo.queueCount = 1;
		queueInfo.pQueuePriorities = &queueProperties;
	}

	// No Extra Features for now
	VkPhysicalDeviceFeatures deviceFeatures{};

	VkDeviceCreateInfo deviceInfo{}; {
		deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		deviceInfo.queueCreateInfoCount = 1;
		deviceInfo.pQueueCreateInfos = &queueInfo;

		deviceInfo.pEnabledFeatures = &deviceFeatures;
	}

	VkDevice device = VK_NULL_HANDLE;
	if (auto const res = vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device); res != VK_SUCCESS)
	{
		std::fprintf(stderr, "Error: can't create logical device\n");
		std::fprintf(stderr, "vkCreateDevice() returned %s\n", lut::to_string(res).c_str());

		return VK_NULL_HANDLE;
	}

	return device;
}
}

namespace
{
VkDebugUtilsMessengerEXT create_debug_messenger(VkInstance instance)
{
	assert(instance != VK_NULL_HANDLE);

	VkDebugUtilsMessengerCreateInfoEXT debugInfo{}; {
		debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

		debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
									VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
								VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
								VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

		debugInfo.pfnUserCallback = &debug_util_callback;
		debugInfo.pUserData = nullptr;
	}

	VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
	if (auto const res = vkCreateDebugUtilsMessengerEXT(instance, &debugInfo, nullptr, &messenger); res != VK_SUCCESS)
	{
		std::fprintf(stderr, "Error: unable to set up debug messenger\n");
		std::fprintf(stderr, "vkCreateDebugUtilsMessengerEXT() returned %s\n", lut::to_string(res).c_str());

		return VK_NULL_HANDLE;
	}

	return messenger;
}
VKAPI_ATTR VkBool32 VKAPI_CALL debug_util_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
												   VkDebugUtilsMessageTypeFlagsEXT type, VkDebugUtilsMessengerCallbackDataEXT
												   const* data, void* userPtr)
{
	std::fprintf(stderr, "%s (%s): %s (%d)\n%s\n--\n", lut::to_string(severity).c_str(), lut::message_type_flags(type).c_str(),
			 	 data->pMessageIdName, data->messageIdNumber, data->pMessage);

	return VK_FALSE;
}
}

//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
