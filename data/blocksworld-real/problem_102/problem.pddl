(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        purple - block
        yellow - block
        blue - block
        red - block
        green - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (on purple orange)
        (on yellow purple)
        (on blue yellow)
        (on red blue)
        (on green red)
        (clear green)
        ; orange -> purple -> yellow -> blue -> red -> green
    )
    (:goal (and (on purple yellow)
        (on blue purple)
        (on orange blue)
        (on red orange)
        (on green red)))
)
    