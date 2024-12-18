#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

// #include "perf.h"
#include "cfu.h"
// #include "cstdio"

//#include "playground_util/print_params.h"

namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  
  //  perf_enable_counter(6);
  
  // 获取参数
  const int32_t input_offset = params.input_offset;
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;
  
  // printf("input_offset: %ld\n", input_offset);
  // printf("stride_width: %d\n", stride_width);
  // printf("stride_height: %d\n", stride_height);
  // printf("dilation_width_factor: %d\n", dilation_width_factor);
  // printf("dilation_height_factor: %d\n", dilation_height_factor);
  // printf("pad_width: %d\n", pad_width);
  // printf("pad_height: %d\n", pad_height);
  // printf("output_offset: %ld\n", output_offset);


  // 设置输出的最小值和最大值
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // printf("output_activation_min: %ld\n", output_activation_min);
  // printf("output_activation_max: %ld\n", output_activation_max);

  // 一致性检查
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // 检查张量的维度
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  // const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  // const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  // printf("input_height: %d\n", input_height);
  // printf("input_width: %d\n", input_width);
  // printf("input_depth: %d\n", input_depth);
  // printf("filter_height: %d\n", filter_height);
  // printf("filter_width: %d\n", filter_width);
  // printf("filter_input_depth: %d\n", filter_input_depth);
  // printf("output_height: %d\n", output_height);
  // printf("output_width: %d\n", output_width);
  // printf("output_depth: %d\n", output_depth);


  // perf_enable_counter(5);
  // 定义 im2col 和 fr2row 数组
  int8_t im2col[1024][1024]; // 根据需要调整大小
  int8_t fr2row[1024][1024];

  // 初始化 im2col
  // for(int i = 0; i < 1024; i++)
  //   for(int j = 0; j < 1024; j++)
  //     im2col[i][j] = -input_offset;
  
  // memset(im2col, -input_offset, sizeof(im2col));


  // // 初始化 fr2row
  // for(int i = 0; i < 1024; i++)
  //   for(int j = 0; j < 1024; j++)
  //     fr2row[i][j] = 0;

  int row_index = 0;
  int col_index = 0;

  // perf_disable_counter(5);

  // perf_enable_counter(0);
  // 图像转换为列
  for (int out_y = 0; out_y < output_height; ++out_y) {
    const int in_y_origin = (out_y * stride_height) - pad_height;
    for (int out_x = 0; out_x < output_width; ++out_x) {
      const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          const int in_y = in_y_origin + filter_y*dilation_height_factor;
          for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
            const int in_x = in_x_origin + filter_x*dilation_width_factor;
            const bool is_point_inside_image = (in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height);
            for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel) {
              row_index = out_y * output_width + out_x;
              col_index = filter_height * filter_width * in_channel + filter_y * filter_width + filter_x;
              if(is_point_inside_image)
                im2col[row_index][col_index] = *((int8_t *)(input_data + Offset(input_shape, 0, in_y, in_x, in_channel)));
              else
                im2col[row_index][col_index] = -input_offset;
            }
          }
      }
    }
  }
  // perf_disable_counter(0);
  // perf_enable_counter(1);

  // 滤波器转换为行
for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
            // 假設 filter_input_depth 已知為 4 的倍數
            int in_channel = 0;
            for (; in_channel <= filter_input_depth - 4; in_channel += 4) {
                // 預先計算一次共用的 row_index
                int common_row_index_0 = filter_height * filter_width * in_channel + filter_y * filter_width + filter_x;
                int common_row_index_1 = common_row_index_0 + (filter_height * filter_width); // in_channel + 1
                int common_row_index_2 = common_row_index_1 + (filter_height * filter_width); // in_channel + 2
                int common_row_index_3 = common_row_index_2 + (filter_height * filter_width); // in_channel + 3
                int common_col_index = out_channel;

                // 連續讀取 4 個 in_channel
                fr2row[common_row_index_0][common_col_index] = *((int8_t*)(filter_data + Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)));
                fr2row[common_row_index_1][common_col_index] = *((int8_t*)(filter_data + Offset(filter_shape, out_channel, filter_y, filter_x, in_channel + 1)));
                fr2row[common_row_index_2][common_col_index] = *((int8_t*)(filter_data + Offset(filter_shape, out_channel, filter_y, filter_x, in_channel + 2)));
                fr2row[common_row_index_3][common_col_index] = *((int8_t*)(filter_data + Offset(filter_shape, out_channel, filter_y, filter_x, in_channel + 3)));
            }

            // 對於不滿 4 的餘數部分再處理一次
            for (; in_channel < filter_input_depth; ++in_channel) {
                 row_index = filter_height * filter_width * in_channel + filter_y * filter_width + filter_x;
                 col_index = out_channel;
                fr2row[row_index][col_index] = *((int8_t*)(filter_data + Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)));
            }
        }
    }
}


  // perf_disable_counter(1);
  // perf_enable_counter(2);

  // 重置 CFU
  cfu_op0(1, 0, 0);
  cfu_op0(18, input_offset, 0); // input_offset

  // printf("input_offset: %lx\n", input_offset);
  // printf("-input_offset: %lx\n", -input_offset);

  // 设置维度 K
  const int K = filter_height * filter_width * input_depth;
  const int N = output_depth;
  const int M = output_height * output_width;
  const int index_bound = M*N;
  cfu_op0(2, K, 0);
  cfu_op0(4,8,0);
  cfu_op0(6,4,0);

  // perf_disable_counter(2);
  // perf_enable_counter(3);
  // printf("output_depth: %d\n", output_depth);
  for(int out_channel = 0; out_channel < output_depth; out_channel += 4){
    // 加载缓冲区 B
    for(int kk = 0; kk < K; kk++){
      uint8_t b[4] = {0, 0, 0, 0};
      for (int c = 0; c < 4; ++c) {
        b[c] = fr2row[kk][out_channel + c];
      }
      // printf("b: %x %x %x %x\n", b[0], b[1], b[2], b[3]);
      uint32_t in_b = ((b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]);

      // 写入全局缓冲区 B
      cfu_op0(10, kk, in_b);

      // print buffer B
      // int32_t ret2 = cfu_op0(11, kk, 0);
      // printf("Set Buffer B set = %lx \t\taddr: %x, \t\tout: %lX\n", in_b , kk, ret2);
    }

    for(int slide = 0; slide < output_height * output_width; slide += 8){
      // 加载缓冲区 A
      for(int kk = 0; kk < K; kk++){
        uint8_t a[4] = {0, 0, 0, 0};
        uint8_t a1[4] = {0, 0, 0, 0};
        // uint8_t a2[4] = {0, 0, 0, 0};
        // uint8_t a3[4] = {0, 0, 0, 0};
        for (int s = 0; s < 4; ++s){
          a[s] = im2col[slide + s][kk];
          a1[s] = im2col[slide + 4 + s][kk];
          // a2[s] = im2col[slide + 8 + s][kk];
          // a3[s] = im2col[slide + 12 + s][kk];
        }
        // printf("a: %x %x %x %x\n", a[0], a[1], a[2], a[3]);
        uint32_t in_a = ((a[0] << 24) | (a[1] << 16) | (a[2] << 8) | a[3]);
        uint32_t in_a1 = ((a1[0] << 24) | (a1[1] << 16) | (a1[2] << 8) | a1[3]);
        // uint32_t in_a2 = ((a2[0] << 24) | (a2[1] << 16) | (a2[2] << 8) | a2[3]);
        // uint32_t in_a3 = ((a3[0] << 24) | (a3[1] << 16) | (a3[2] << 8) | a3[3]);

        // 写入全局缓冲区 A
        cfu_op0(8, kk, in_a);
        cfu_op0(8, kk + K, in_a1);
        // cfu_op0(8, kk + 2*K, in_a2);
        // cfu_op0(8, kk + 3*K, in_a3);

        // int32_t ret = cfu_op0(9, kk, 0);
        // printf("Set Buffer A = %lx , \t\taddr: %x, \t\tout: %lX\n", in_a , kk, ret);
      }

      // 启动 CFU
      cfu_op0(12, 0, 0);

      int tmp_slide = slide;
      for (int s = 0; s < 8; ++s) {
        // 每個通道對應固定的數值直接寫入
        int32_t acc_value = cfu_op0(17, s, 0);
        if (bias_data) acc_value += bias_data[out_channel + 0];
        acc_value = MultiplyByQuantizedMultiplier(acc_value, output_multiplier[out_channel + 0], output_shift[out_channel + 0]);
        acc_value += output_offset;
        acc_value = std::max(acc_value, output_activation_min);
        acc_value = std::min(acc_value, output_activation_max);
        int output_index = Offset(output_shape, 0, (tmp_slide + s%4) / output_width, (tmp_slide + s%4) % output_width, out_channel + 0);
        if (output_index < index_bound) {
            output_data[output_index] = static_cast<int8_t>(acc_value);
        }

        acc_value = cfu_op0(16, s, 0);
        if (bias_data) acc_value += bias_data[out_channel + 1];
        acc_value = MultiplyByQuantizedMultiplier(acc_value, output_multiplier[out_channel + 1], output_shift[out_channel + 1]);
        acc_value += output_offset;
        acc_value = std::max(acc_value, output_activation_min);
        acc_value = std::min(acc_value, output_activation_max);
        output_index = Offset(output_shape, 0, (tmp_slide + s%4) / output_width, (tmp_slide + s%4) % output_width, out_channel + 1);
        if (output_index < index_bound) {
            output_data[output_index] = static_cast<int8_t>(acc_value);
        }

        acc_value = cfu_op0(15, s, 0);
        if (bias_data) acc_value += bias_data[out_channel + 2];
        acc_value = MultiplyByQuantizedMultiplier(acc_value, output_multiplier[out_channel + 2], output_shift[out_channel + 2]);
        acc_value += output_offset;
        acc_value = std::max(acc_value, output_activation_min);
        acc_value = std::min(acc_value, output_activation_max);
        output_index = Offset(output_shape, 0, (tmp_slide + s%4) / output_width, (tmp_slide + s%4) % output_width, out_channel + 2);
        if (output_index < index_bound) {
            output_data[output_index] = static_cast<int8_t>(acc_value);
        }

        acc_value = cfu_op0(14, s, 0);
        if (bias_data) acc_value += bias_data[out_channel + 3];
        acc_value = MultiplyByQuantizedMultiplier(acc_value, output_multiplier[out_channel + 3], output_shift[out_channel + 3]);
        acc_value += output_offset;
        acc_value = std::max(acc_value, output_activation_min);
        acc_value = std::min(acc_value, output_activation_max);
        output_index = Offset(output_shape, 0, (tmp_slide + s%4) / output_width, (tmp_slide + s%4) % output_width, out_channel + 3);
        if (output_index < index_bound) {
            output_data[output_index] = static_cast<int8_t>(acc_value);
        }

        if(s%4 == 3) tmp_slide += 4;
      }


    }
  }
  // perf_disable_counter(3);
  // perf_disable_counter(6);
}



inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}


}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_