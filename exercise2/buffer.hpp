#pragma once

#include <volk/volk.h>

class Buffer final
{
	public:
		Buffer() noexcept, ~Buffer();

		explicit Buffer( VkDevice, VkBuffer = VK_NULL_HANDLE, VkDeviceMemory = VK_NULL_HANDLE ) noexcept;

		Buffer( Buffer const& ) = delete;
		Buffer& operator= (Buffer const&) = delete;

		Buffer( Buffer&& ) noexcept;
		Buffer& operator = (Buffer&&) noexcept;

	public:
		VkBuffer buffer = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;

	private:
		VkDevice mDevice = VK_NULL_HANDLE;
};

