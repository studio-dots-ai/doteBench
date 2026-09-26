form Worker
    sentence control_dir
endform
request$ = control_dir$ + "/request.ready"
response$ = control_dir$ + "/response.txt"
done$ = control_dir$ + "/done.txt"
stop$ = control_dir$ + "/stop.ready"
while not fileReadable (stop$)
    if fileReadable (request$)
        lines$# = readLinesFromFile$# (request$)
        deleteFile: request$
        wav$ = lines$# [2]
        step = number (lines$# [3])
        floor = number (lines$# [4])
        ceiling = number (lines$# [5])
        sound = Read from file: wav$
        selectObject: sound
        pitch = To Pitch (ac): step, floor, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, ceiling
        selectObject: pitch
        n = Get number of frames
        writeInfoLine: "frames ", n
        for i from 1 to n
            t = Get time from frame: i
            f = Get value in frame: i, "Hertz"
            if f = undefined
                appendInfoLine: fixed$ (t, 12), " 0"
            else
                appendInfoLine: fixed$ (t, 12), " ", fixed$ (f, 12)
            endif
        endfor
        writeFile: response$, info$ ( )
        removeObject: pitch, sound
        writeFileLine: done$, lines$# [1]
    else
        dummy = sleep (0.01)
    endif
endwhile
