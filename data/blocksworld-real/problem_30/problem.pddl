(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        blue - block
        green - block
        purple - block
        yellow - block
        red - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable blue)
        (on green blue)
        (on purple green)
        (on yellow purple)
        (on red yellow)
        (on orange red)
        (clear orange)
        ; blue -> green -> purple -> yellow -> red -> orange
    )
    (:goal (and (on yellow red)
        (on blue yellow)
        (on green blue)
        (on purple green)
        (on orange purple)))
)
    