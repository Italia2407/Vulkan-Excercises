#include <volk/volk.h>

#include <tuple>
#include <limits>
#include <vector>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"
namespace lut = labutils;

namespace
{
	namespace cfg
	{
		// Compiled shader code for the graphics pipeline
		// See sources in exercise3/shaders/*. 
#		define SHADERDIR_ "assets/exercise3/shaders/"
		constexpr char const* kVertShaderPath = SHADERDIR_ "triangle.vert.spv";
		constexpr char const* kFragShaderPath = SHADERDIR_ "triangle.frag.spv";
#		undef SHADERDIR_
	}

	// GLFW callbacks
	void glfw_callback_key_press( GLFWwindow*, int, int, int, int );

	// Helpers:
	lut::RenderPass create_render_pass( lut::VulkanWindow const& );

	lut::PipelineLayout create_triangle_pipeline_layout( lut::VulkanContext const& );
	lut::Pipeline create_triangle_pipeline( lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout ); // VulkanWindow or VulkanContext?

	void create_swapchain_framebuffers( 
		lut::VulkanWindow const&, 
		VkRenderPass,
		std::vector<lut::Framebuffer>&
	);

	void record_commands( 
		VkCommandBuffer,
		VkRenderPass,
		VkFramebuffer,
		VkPipeline,
		VkExtent2D const&
	);
	void submit_commands(
		lut::VulkanContext const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);
}

int main() try
{
	// Create our Vulkan Window
	lut::VulkanWindow window = lut::make_vulkan_window();

	// Configure the GLFW window
	glfwSetKeyCallback( window.window, &glfw_callback_key_press );

	// Intialize resources
	lut::RenderPass renderPass = create_render_pass( window );

	lut::PipelineLayout pipeLayout = create_triangle_pipeline_layout( window );
	lut::Pipeline pipe = create_triangle_pipeline( window, renderPass.handle, pipeLayout.handle );

	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers( window, renderPass.handle, framebuffers );


	lut::CommandPool cpool = lut::create_command_pool( window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT );

	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;
	
	for( std::size_t i = 0; i < framebuffers.size(); ++i )
	{
		cbuffers.emplace_back( lut::alloc_command_buffer( window, cpool.handle ) );
		cbfences.emplace_back( lut::create_fence( window, VK_FENCE_CREATE_SIGNALED_BIT ) );
	}

	lut::Semaphore imageAvailable = lut::create_semaphore( window );
	lut::Semaphore renderFinished = lut::create_semaphore( window );

	// Application main loop
	bool recreateSwapchain = false;

	while( !glfwWindowShouldClose( window.window ) )
	{
		// Let GLFW process events.
		// glfwPollEvents() checks for events, processes them. If there are no
		// events, it will return immediately. Alternatively, glfwWaitEvents()
		// will wait for any event to occur, process it, and only return at
		// that point. The former is useful for applications where you want to
		// render as fast as possible, whereas the latter is useful for
		// input-driven applications, where redrawing is only needed in
		// reaction to user input (or similar).
		glfwPollEvents(); // or: glfwWaitEvents()

		// Recreate swap chain?
		if( recreateSwapchain )
		{
			vkDeviceWaitIdle(window.device);

			auto const changes = lut::recreate_swapchain(window);

			if (changes.changedFormat)
				renderPass = create_render_pass(window);

			framebuffers.clear();
			create_swapchain_framebuffers(window, renderPass.handle, framebuffers);

			if (changes.changedSize)
				pipe = create_triangle_pipeline(window, renderPass.handle, pipeLayout.handle);
			
			recreateSwapchain = false;
			continue;
		}

		std::uint32_t imageIndex = 0;
		auto const aquireRes = vkAcquireNextImageKHR(
			window.device,
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable.handle,
			VK_NULL_HANDLE,
			&imageIndex
		);

		if (aquireRes == VK_SUBOPTIMAL_KHR || aquireRes == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreateSwapchain = true;
			continue;
		}
		else if (aquireRes != VK_SUCCESS)
		{
			throw lut::Error("Unable to Acquire enxt Swapchain Image\n"
				"vkAcquireNextImageKHR() Returned %s", lut::to_string(aquireRes).c_str());
		}

		assert(std::size_t(imageIndex) < cbfences.size());

		if (auto const res = vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); res != VK_SUCCESS)
		{
			throw lut::Error("Unable to Wait for Command Buffer Fence %u\n"
				"vkWaitForFences() Returned %s", imageIndex, lut::to_string(res).c_str());
		}

		if (auto const res = vkResetFences(window.device, 1, &cbfences[imageIndex].handle); res != VK_SUCCESS)
		{
			throw lut::Error("Unable to Reset Command Buffer Fence %u\n"
				"vkResetFences() Returned %s", imageIndex, lut::to_string(res).c_str());
		}
		
		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());

		record_commands(
			cbuffers[imageIndex],
			renderPass.handle,
			framebuffers[imageIndex].handle,
			pipe.handle,
			window.swapchainExtent
		);
		submit_commands(
			window,
			cbuffers[imageIndex],
			cbfences[imageIndex].handle,
			imageAvailable.handle,
			renderFinished.handle
		);

		VkPresentInfoKHR presentInfo{}; {
			presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

			presentInfo.waitSemaphoreCount = 1;
			presentInfo.pWaitSemaphores = &renderFinished.handle;

			presentInfo.swapchainCount = 1;
			presentInfo.pSwapchains = &window.swapchain;

			presentInfo.pImageIndices = &imageIndex;
			presentInfo.pResults = nullptr;
		}

		auto const presentRes = vkQueuePresentKHR(window.presentQueue, &presentInfo);
		if (presentRes == VK_SUBOPTIMAL_KHR || presentRes == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreateSwapchain = true;
		}
		else if (presentRes != VK_SUCCESS)
		{
			throw lut::Error("Unable to Present Swapchain Image %u\n"
				"vkQueuePresentKHR() Returned %s", imageIndex, lut::to_string(presentRes).c_str());
		}
	}
	
	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.
	vkDeviceWaitIdle( window.device );

	return 0;
}
catch( std::exception const& eErr )
{
	std::fprintf( stderr, "Exception: %s\n", eErr.what() );
	return 1;
}

namespace
{
void glfw_callback_key_press( GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/ )
{
	if( GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction )
	{
		glfwSetWindowShouldClose( aWindow, GLFW_TRUE );
	}
}
}

namespace
{
lut::RenderPass create_render_pass( lut::VulkanWindow const& aWindow )
{
	VkAttachmentDescription attachments[1]{}; {
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	}

	VkAttachmentReference subpassAttachments[1]{}; {
		subpassAttachments[0].attachment = 0;
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	}
	VkSubpassDescription subpasses[1]{}; {
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;
	}

	VkRenderPassCreateInfo renderPassInfo{}; {
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = attachments;

		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = subpasses;

		renderPassInfo.dependencyCount = 0;
		renderPassInfo.pDependencies = nullptr;
	}

	VkRenderPass renderPass = VK_NULL_HANDLE;
	if (auto const res = vkCreateRenderPass(aWindow.device, &renderPassInfo, nullptr, &renderPass); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Render Pass\n"
			"vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
	}

	return lut::RenderPass(aWindow.device, renderPass);
}

lut::PipelineLayout create_triangle_pipeline_layout( lut::VulkanContext const& aContext )
{
	VkPipelineLayoutCreateInfo pipelineLayoutInfo{}; {
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

		pipelineLayoutInfo.setLayoutCount = 0;
		pipelineLayoutInfo.pSetLayouts = nullptr;

		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges = nullptr;
	}

	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	if (auto const res = vkCreatePipelineLayout(aContext.device, &pipelineLayoutInfo, nullptr, &pipelineLayout); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Pipeline Layout\n"
			"vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
	}

	return lut::PipelineLayout(aContext.device, pipelineLayout);
}
lut::Pipeline create_triangle_pipeline( lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout )
{
	lut::ShaderModule vertShader = lut::load_shader_module(aWindow, cfg::kVertShaderPath);
	lut::ShaderModule fragShader = lut::load_shader_module(aWindow, cfg::kFragShaderPath);

	VkPipelineShaderStageCreateInfo shaderStagesInfo[2]{}; {
		shaderStagesInfo[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStagesInfo[0].pName = "main";

		shaderStagesInfo[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderStagesInfo[0].module = vertShader.handle;

		shaderStagesInfo[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStagesInfo[1].pName = "main";

		shaderStagesInfo[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderStagesInfo[1].module = fragShader.handle;
	}

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{}; {
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	}
	VkPipelineInputAssemblyStateCreateInfo assemblyStateInfo{}; {
		assemblyStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;

		assemblyStateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyStateInfo.primitiveRestartEnable = VK_FALSE;
	}
	
	VkViewport viewport{}; {
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);

		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
	}
	VkRect2D scissor{}; {
		scissor.offset = VkOffset2D{0, 0};
		scissor.extent = aWindow.swapchainExtent;
	}
	VkPipelineViewportStateCreateInfo viewportStateInfo{}; {
		viewportStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;

		viewportStateInfo.viewportCount = 1;
		viewportStateInfo.pViewports = &viewport;

		viewportStateInfo.scissorCount = 1;
		viewportStateInfo.pScissors = &scissor;
	}
	
	VkPipelineRasterizationStateCreateInfo rasterizationStateInfo{}; {
		rasterizationStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;

		rasterizationStateInfo.depthClampEnable = VK_FALSE;
		rasterizationStateInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterizationStateInfo.depthBiasEnable = VK_FALSE;

		rasterizationStateInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizationStateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizationStateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

		rasterizationStateInfo.lineWidth = 1.0f;
	}
	
	VkPipelineMultisampleStateCreateInfo multisamplingStateInfo{}; {
		multisamplingStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;

		multisamplingStateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	}
	
	VkPipelineColorBlendAttachmentState colourBlendAttachmentStates[1]{}; {
		colourBlendAttachmentStates[0].blendEnable = VK_FALSE;
		colourBlendAttachmentStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	}
	VkPipelineColorBlendStateCreateInfo colourBlendStateInfo{}; {
		colourBlendStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

		colourBlendStateInfo.logicOpEnable = VK_FALSE;

		colourBlendStateInfo.attachmentCount = 1;
		colourBlendStateInfo.pAttachments = colourBlendAttachmentStates;
	}
	
	VkGraphicsPipelineCreateInfo graphicsPipelineInfo{}; {
		graphicsPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		graphicsPipelineInfo.stageCount = 2;
		graphicsPipelineInfo.pStages = shaderStagesInfo;

		graphicsPipelineInfo.pVertexInputState = &vertexInputInfo;
		graphicsPipelineInfo.pInputAssemblyState = &assemblyStateInfo;
		graphicsPipelineInfo.pTessellationState = nullptr;
		graphicsPipelineInfo.pViewportState = &viewportStateInfo;
		graphicsPipelineInfo.pRasterizationState = &rasterizationStateInfo;
		graphicsPipelineInfo.pMultisampleState = &multisamplingStateInfo;
		graphicsPipelineInfo.pDepthStencilState = nullptr;
		graphicsPipelineInfo.pColorBlendState = &colourBlendStateInfo;
		graphicsPipelineInfo.pDynamicState = nullptr;

		graphicsPipelineInfo.layout = aPipelineLayout;
		graphicsPipelineInfo.renderPass = aRenderPass;
		graphicsPipelineInfo.subpass = 0;
	}
	
	VkPipeline graphicsPipeline = VK_NULL_HANDLE;
	if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &graphicsPipelineInfo, nullptr, &graphicsPipeline); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Create Graphics Pipeline\n"
			"vkCreateGraphicsPipeline() returned %s", lut::to_string(res).c_str());
	}
	
	return lut::Pipeline(aWindow.device, graphicsPipeline);
}

void create_swapchain_framebuffers( lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers )
{
	assert( aFramebuffers.empty() );

	for (std::size_t i = 0; i < aWindow.swapViews.size(); i++)
	{
		VkImageView attachments[1] = {
			aWindow.swapViews[i]
		};

		VkFramebufferCreateInfo framebufferInfo{}; {
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;

			framebufferInfo.flags = 0;
			framebufferInfo.renderPass = aRenderPass;

			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments = attachments;

			framebufferInfo.width = aWindow.swapchainExtent.width;
			framebufferInfo.height = aWindow.swapchainExtent.height;

			framebufferInfo.layers = 1;
		}

		VkFramebuffer framebuffer = VK_NULL_HANDLE;
		if (auto const res = vkCreateFramebuffer(aWindow.device, &framebufferInfo, nullptr, &framebuffer); res != VK_SUCCESS)
		{
			throw lut::Error("Unable to Create Framebuffer for Swapchain Image %zu\n"
				"vkCreateFramebuffer() Returned %s", i, lut::to_string(res).c_str());
		}
		aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, framebuffer));
	}

	assert( aWindow.swapViews.size() == aFramebuffers.size() );
}

void record_commands( VkCommandBuffer aCmdBuff, VkRenderPass aRenderPass, VkFramebuffer aFramebuffer, VkPipeline aGraphicsPipe, VkExtent2D const& aImageExtent )
{
	// Begin Recording Commands
	VkCommandBufferBeginInfo commandBufferBeginInfo{}; {
		commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		commandBufferBeginInfo.pInheritanceInfo = nullptr;
	}

	if (auto const res = vkBeginCommandBuffer(aCmdBuff, &commandBufferBeginInfo); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Begin Recording Command Buffer\n"
			"vkBeginCommandBuffer() Returned %s", lut::to_string(res).c_str());
	}

	// Begin the Render Pass
	VkClearValue clearValues[1]{}; {
		clearValues[0].color.float32[0] = 0.1f;
		clearValues[0].color.float32[1] = 0.1f;
		clearValues[0].color.float32[2] = 0.1f;
		clearValues[0].color.float32[3] = 1.0f;
	}
	
	VkRenderPassBeginInfo renderPassInfo{}; {
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;

		renderPassInfo.renderPass = aRenderPass;
		renderPassInfo.framebuffer = aFramebuffer;

		renderPassInfo.renderArea.offset = VkOffset2D{0, 0};
		renderPassInfo.renderArea.extent = aImageExtent;

		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = clearValues;
	}

	vkCmdBeginRenderPass(aCmdBuff, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	// Begin Drawing with our Graphics Pipeline
	vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsPipe);

	// Draw a Triangle
	//vkCmdDraw(aCmdBuff, 3, 1, 0, 0);
	//vkCmdDraw(aCmdBuff, 3, 1, 3, 1);
	vkCmdDraw(aCmdBuff, 6, 2, 0, 0);

	// End the Render Pass
	vkCmdEndRenderPass(aCmdBuff);

	// End Command Recording
	if (auto const res = vkEndCommandBuffer(aCmdBuff); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to End Recording Command Buffer\n"
			"vkEndCommandBuffer() Returned %s", lut::to_string(res).c_str());
	}
}
void submit_commands( lut::VulkanContext const& aContext, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore )
{
	VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	VkSubmitInfo submitInfo{}; {
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;

		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = &waitPipelineStages;

		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;
	}

	if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, aFence); res != VK_SUCCESS)
	{
		throw lut::Error("Unable to Submit Command Buffer to Queue\n"
			"vkQueueSubmit() Returned %s", lut::to_string(res).c_str());
	}
}
}

//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
