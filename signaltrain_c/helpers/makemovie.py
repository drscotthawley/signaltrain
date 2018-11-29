def makemovie(datagen, model, batch_size):
    print("\n\nMaking movies")
    for sig_type in [0,4]:
        print()
        x_val_cuda2, y_val_cuda2, knobs_val_cuda2 = datagen_movie.new(chooser=sig_type)
        frame = 0
        intervals = 7
        for t in np.linspace(-0.5,0.5,intervals):          # threshold
            for r in np.linspace(-0.5,0.5,intervals):      # ratio
                for a in np.linspace(-0.5,0.5,intervals):  # attack
                    frame += 1
                    print(f'\rframe = {frame}/{intervals**3-1}.   ',end="")
                    knobs = np.array([t, r, a])
                    x_val_cuda2, y_val_cuda2, knobs_val_cuda2 = datagen_movie.new(knobs=knobs, recyc_x=True, chooser=sig_type)
                    x_val_hat2, mag_val2, mag_val_hat2 = model.forward(x_val_cuda2, knobs_val_cuda2)
                    loss_val2 = calc_loss(x_val_hat2,y_val_cuda2,mag_val2,objective,batch_size=batch_size)

                    framename = f'movie{sig_type}_{frame:04}.png'
                    print(f'Saving {framename}           ',end="")
                    plot_valdata(x_val_cuda2, knobs_val_cuda2, y_val_cuda2, x_val_hat2, knob_ranges, epoch, loss_val, filename=framename)
        shellcmd = f'rm -f movie{sig_type}.mp4; ffmpeg -framerate 10 -i movie{sig_type}_%04d.png -c:v libx264 -vf format=yuv420p movie{sig_type}.mp4; rm -f movie{sig_type}_*.png'
        p = call(shellcmd, stdout=PIPE, shell=True)
    return
