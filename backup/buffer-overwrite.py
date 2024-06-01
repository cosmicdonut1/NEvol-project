# Save the buffer periodically
# current_time = time.time()
# if current_time - last_save_time >= save_interval:
#     filename = os.path.join(save_path, f"buffer_{int(current_time)}.npz")
#     buffer.save_buffer(filename)
#     buffer.clear_buffer()
#     last_save_time = current_time
# print("-------------------Buffer print---------------------")
# buffer.print_buffer()
# buffer.print_buffer_shape()