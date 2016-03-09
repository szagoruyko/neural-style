require 'torch'
require 'nn'
require 'image'
require 'optim'

require 'loadcaffe'

local autograd = require 'autograd'

--------------------------------------------------------------------------------

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style_image', 'examples/inputs/seated-nude.jpg',
           'Style target image')
cmd:option('-style_blend_weights', 'nil')
cmd:option('-content_image', 'examples/inputs/tubingen.jpg',
           'Content target image')
cmd:option('-image_size', 512, 'Maximum height / width of generated image')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')

-- Optimization options
cmd:option('-content_weight', 5e0)
cmd:option('-style_weight', 1e2)
cmd:option('-tv_weight', 1e-3)
cmd:option('-num_iterations', 1000)
cmd:option('-normalize_gradients', false)
cmd:option('-init', 'random', 'random|image')
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-learning_rate', 1e1)

-- Output options
cmd:option('-print_iter', 50)
cmd:option('-save_iter', 100)
cmd:option('-output_image', 'out.png')

-- Other options
cmd:option('-style_scale', 1.0)
cmd:option('-pooling', 'max', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-cudnn_autotune', false)
cmd:option('-seed', -1)

cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')

local function main(params)
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(params.gpu + 1)
    else
      require 'clnn'
      require 'cltorch'
      cltorch.setDevice(params.gpu + 1)
    end
  else
    params.backend = 'nn'
  end

  local backend = params.backend == 'cudnn' and 'cudnn' or 'nn'
  if params.backend == 'cudnn' then
    require 'cudnn'
    autograd.cudnn = autograd.functionalize('cudnn')
    if params.cudnn_autotune then
      cudnn.benchmark = true
    end
  end

  local function cast(m)
     m = m:float()
     if params.gpu >= 0 then
        if params.backend ~= 'clnn' then
           m = m:cuda()
        else
           m = m:cl()
        end
     end
     return m
  end
  
  local loadcaffe_backend = params.backend
  if params.backend == 'clnn' then loadcaffe_backend = 'nn' end
  local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):float()
  cast(cnn)
  
  local content_image = image.load(params.content_image, 3)
  content_image = image.scale(content_image, params.image_size, 'bilinear')
  local content_image_caffe = preprocess(content_image):float()
  
  local style_size = math.ceil(params.style_scale * params.image_size)
  local style_image_list = params.style_image:split(',')
  local style_images_caffe = {}
  for _, img_path in ipairs(style_image_list) do
    local img = image.load(img_path, 3)
    img = image.scale(img, style_size, 'bilinear')
    local img_caffe = preprocess(img):float()
    table.insert(style_images_caffe, img_caffe)
  end

  -- Handle style blending weights for multiple style inputs
  local style_blend_weights = nil
  if params.style_blend_weights == 'nil' then
    -- Style blending not specified, so use equal weighting
    style_blend_weights = {}
    for i = 1, #style_image_list do
      table.insert(style_blend_weights, 1.0)
    end
  else
    style_blend_weights = params.style_blend_weights:split(',')
    assert(#style_blend_weights == #style_image_list,
      '-style_blend_weights and -style_images must have the same number of elements')
  end
  -- Normalize the style blending weights so they sum to 1
  local style_blend_sum = 0
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = tonumber(style_blend_weights[i])
    style_blend_sum = style_blend_sum + style_blend_weights[i]
  end
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = style_blend_weights[i] / style_blend_sum
  end
  

  content_image_caffe = cast(content_image_caffe)
  for i = 1, #style_images_caffe do
     style_images_caffe[i] = cast(style_images_caffe[i])
  end
  
  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")

  -- have to disable accGradParameters manually, autograd calls backward which calls both
  -- updateGradInput and accGradParameters
  cast(cnn)
  do
     local b = backend == 'cudnn' and cudnn or nn
     b.SpatialConvolution.accGradParameters = function() end
  end

  -- Set up the network, inserting style and content loss modules
  local net_params,grad_layers = {}, {}
  local next_content_idx, next_style_idx = 1, 1
  local tvloss = params.tv_weight > 0 and cast(nn.TVLoss(params.tv_weight))
  local autograd_backend = autograd[backend]
  for i,v in ipairs(cnn.modules) do
     if next_content_idx <= #content_layers or next_style_idx <= #style_layers then
        if torch.type(v):find'SpatialConvolution' then
           local c, params = autograd_backend.SpatialConvolution(v.nInputPlane, v.nOutputPlane, v.kW, v.kH, v.dW, v.dH, v.padW, v.padH)
           table.insert(grad_layers, {layer = c, params = #net_params + 1})
           table.insert(net_params, {v.weight, v.bias})
        elseif torch.type(v):find'ReLU' then
           local c = autograd_backend.ReLU(true)
           table.insert(grad_layers, {layer = c})
        elseif torch.type(v):find'MaxPooling' then
           local pooling = params.pooling == 'avg' and 'SpatialAveragePooling' or 'SpatialMaxPooling'
           local c = autograd_backend[pooling](v.kW, v.kH, v.dW, v.dH, v.padW, v.padH)
           table.insert(grad_layers, {layer = c})
        elseif torch.type(v):find'AveragePooling' then
           local c = autograd_backend.SpatialAveragePooling(v.kW, v.kH, v.dW, v.dH, v.padW, v.padH)
           table.insert(grad_layers, {layer = c})
        end

        if v.name == content_layers[next_content_idx] then
           table.insert(grad_layers, {type = 'content'})
           next_content_idx = next_content_idx + 1
        end
        if v.name == style_layers[next_style_idx] then
           table.insert(grad_layers, {type = 'style'})
           next_style_idx = next_style_idx + 1
        end
     end
  end

  local function gram(x)
     x = x:view(x:size(1),-1)
     return x * torch.transpose(x,1,2)
  end

  -- this is our main forward function
  local function f(layers, input)
     local output = input
     local style_outputs, content_outputs = {}, {}
     for i,v in ipairs(grad_layers) do
        if v.type == 'style' then
           table.insert(style_outputs, output)
        elseif v.type == 'content' then
           table.insert(content_outputs, output)
        elseif v.params then
           output = v.layer(layers[v.params], output)
        else
           output = v.layer(output)
        end
     end
     return {style_outputs, content_outputs, output}
  end
  
  -- this is our main function that computes the loss
  -- here param is input image, layers are weights table
  local function predict(param, layers, target)
     local outputs = f(layers, param)
     local loss = 0
     local style_outputs, content_outputs = outputs[1], outputs[2]
     local style_targets, content_targets = target[1], target[2]
     for i,v in ipairs(style_outputs) do
        local n = torch.nElement(v)
        local gram_n = torch.nElement(style_targets[i])
        loss = loss + params.style_weight * autograd.loss.leastSquares(gram(v) / n, style_targets[i]) / gram_n
     end
     for i,v in ipairs(content_outputs) do
        loss = loss + params.content_weight * autograd.loss.leastSquares(v, content_targets[i]) / torch.nElement(v)
     end
     return loss
  end


  local g = autograd(predict, {optimize = true})


  -- compute targets
  local style_targets = {}
  for i,v in ipairs(style_images_caffe) do
     for j,u in ipairs(f(net_params, v)[1]) do
        local target = style_targets[j]
        if i == 1 then
           target = gram(u):zero()
        end
        target = target + gram(u) * style_blend_weights[i] / u:numel() 
        style_targets[j] = target
     end
  end
  local content_targets = {}
  for j,u in ipairs(f(net_params, content_image_caffe)[2]) do
     content_targets[j] = u:clone()
  end
  local targets = {style_targets, content_targets}

  -- Initialize the image
  if params.seed >= 0 then
    torch.manualSeed(params.seed)
  end
  local img = nil
  if params.init == 'random' then
    img = torch.randn(content_image:size()):float():mul(0.001)
  elseif params.init == 'image' then
    img = content_image_caffe:clone():float()
  else
    error('Invalid init type')
  end
  img = cast(img)

  -- Declaring this here lets us access it in maybe_print
  local optim_state = nil
  if params.optimizer == 'lbfgs' then
    optim_state = {
      maxIter = params.num_iterations,
      verbose=true,
    }
  elseif params.optimizer == 'adam' then
    optim_state = {
      learningRate = params.learning_rate,
    }
  else
    error(string.format('Unrecognized optimizer "%s"', params.optimizer))
  end

  local function maybe_print(t, loss)
    local verbose = (params.print_iter > 0 and t % params.print_iter == 0)
    if verbose then
      print(string.format('Iteration %d / %d', t, params.num_iterations))
      print(string.format('  Total loss: %f', loss))
    end
  end

  local function maybe_save(t)
    local should_save = params.save_iter > 0 and t % params.save_iter == 0
    should_save = should_save or t == params.num_iterations
    if should_save then
      local disp = deprocess(img:double())
      disp = image.minmax{tensor=disp, min=0, max=1}
      local filename = build_filename(params.output_image, t)
      if t == params.num_iterations then
        filename = params.output_image
      end
      image.save(filename, disp)
    end
  end

  -- Function to evaluate loss and gradient. We run the net forward and
  -- backward to get the gradient, and sum up losses from the loss modules.
  -- optim.lbfgs internally handles iteration and calls this fucntion many
  -- times, so we manually count the number of iterations to handle printing
  -- and saving intermediate results.
  local num_calls = 0
  local function feval(x)
    num_calls = num_calls + 1

    local grad, loss = g(x, net_params, targets)
    if tvloss then grad = tvloss:updateGradInput(x,grad) end
    
    maybe_print(num_calls, loss)
    maybe_save(num_calls)

    return loss, grad:view(grad:nElement())
  end

  -- Run optimization.
  if params.optimizer == 'lbfgs' then
    print('Running optimization with L-BFGS')
    local x, losses = optim.lbfgs(feval, img, optim_state)
  elseif params.optimizer == 'adam' then
    print('Running optimization with ADAM')
    for t = 1, params.num_iterations do
      local x, losses = optim.adam(feval, img, optim_state)
    end
  end
end
  

function build_filename(output_image, iteration)
  local ext = paths.extname(output_image)
  local basename = paths.basename(output_image, ext)
  local directory = paths.dirname(output_image)
  return string.format('%s/%s_%d.%s',directory, basename, iteration, ext)
end


-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end


-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end


local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
   parent.__init(self)
   self.strength = strength
   self.x_diff = torch.Tensor()
   self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
   self.output:set(input)
   return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   local C, H, W = input:size(1), input:size(2), input:size(3)
   self.x_diff:resize(3, H - 1, W - 1)
   self.y_diff:resize(3, H - 1, W - 1)
   self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
   self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
   self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
   self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
   self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
   self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
   self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
   self.gradInput:mul(self.strength)
   self.gradInput:add(gradOutput)
   return self.gradInput
end

local params = cmd:parse(arg)
main(params)
