(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        red - block
        orange - block
        blue - block
        yellow - block
        green - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable red)
        (on orange red)
        (on blue orange)
        (on yellow blue)
        (on green yellow)
        (on purple green)
        (clear purple)
        ; red -> orange -> blue -> yellow -> green -> purple
    )
    (:goal (and (on yellow orange)
        (on purple green)
        (on red blue)))
)
    