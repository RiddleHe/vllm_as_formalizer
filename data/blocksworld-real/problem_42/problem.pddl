(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        blue - block
        orange - block
        red - block
        purple - block
        yellow - block
        green - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable blue)
        (on orange blue)
        (on red orange)
        (on purple red)
        (on yellow purple)
        (on green yellow)
        (clear green)
        ; blue -> orange -> red -> purple -> yellow -> green
    )
    (:goal (and (on purple green)
        (on orange red)
        (on blue yellow)))
)
    