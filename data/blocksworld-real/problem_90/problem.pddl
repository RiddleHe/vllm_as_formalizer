(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        purple - block
        red - block
        green - block
        orange - block
        yellow - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable purple)
        (on red purple)
        (on green red)
        (on orange green)
        (on yellow orange)
        (on blue yellow)
        (clear blue)
        ; purple -> red -> green -> orange -> yellow -> blue
    )
    (:goal (and (on red blue)
        (on orange yellow)
        (on purple green)))
)
    